import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";

import { z } from "zod";

import type {
  ExistingModel,
  SyncProvider,
  SyncedBaseModel,
  SyncedFullModel,
  SyncedModel,
} from "../index.js";
import { buildOpenRouterModel, type OpenRouterModel } from "./openrouter.js";

const API_ENDPOINT = "https://router.requesty.ai/v1/models";
const MODELS_DIR = path.join(import.meta.dirname, "..", "..", "..", "..", "..", "models");
const TOKENS_PER_MILLION = 1_000_000;
const PRICE_DECIMALS = 1_000_000;
const REASONING_EFFORTS = ["none", "low", "medium", "high", "max"] as const;

const PricingBand = z
  .object({
    prompt_tokens_threshold: z.number(),
    input_price: z.number().optional(),
    output_price: z.number().optional(),
    cached_price: z.number().optional(),
    caching_price: z.number().optional(),
  })
  .passthrough();

export const RequestyModel = z
  .object({
    id: z.string().min(1),
    created: z.number(),
    context_window: z.number(),
    max_output_tokens: z.number(),
    input_price: z.number(),
    output_price: z.number(),
    cached_price: z.number().optional(),
    caching_price: z.number().optional(),
    pricing: z.array(PricingBand).optional(),
    supports_vision: z.boolean().optional(),
    supports_reasoning: z.boolean().optional(),
    supports_tool_calling: z.boolean().optional(),
    supports_output_json_schema: z.boolean().optional(),
  })
  .passthrough();

export const RequestyResponse = z
  .object({
    object: z.literal("list"),
    data: z.array(RequestyModel),
  })
  .passthrough();

const BaseMetadata = z
  .object({
    reasoning: z.boolean().optional(),
    temperature: z.boolean().optional(),
    tool_call: z.boolean().optional(),
    structured_output: z.boolean().optional(),
    open_weights: z.boolean().optional(),
    limit: z.object({ context: z.number(), output: z.number().optional() }).passthrough(),
    modalities: z
      .object({ input: z.array(z.string()), output: z.array(z.string()) })
      .passthrough(),
  })
  .passthrough();

export type RequestyModel = z.infer<typeof RequestyModel>;
type BaseMetadata = z.infer<typeof BaseMetadata>;

export const requesty = {
  id: "requesty",
  name: "Requesty",
  modelsDir: "providers/requesty/models",
  sourceID: (model) => model.id,
  skippedNotice: (ids) => [
    `${ids.length} Requesty routes have no \`models/\` metadata entry yet: ${ids.join(", ")}`,
  ],
  async fetchModels() {
    const response = await fetch(API_ENDPOINT);
    if (!response.ok) {
      throw new Error(`Requesty request failed: ${response.status} ${response.statusText}`);
    }
    return response.json();
  },
  parseModels(raw) {
    return RequestyResponse.parse(raw).data;
  },
  translateModel(model, context) {
    const baseModel = resolveRequestyBaseModel(model.id);
    // A route with no metadata to inherit keeps whatever is authored for it.
    if (baseModel === undefined) {
      const authored = context.authored(model.id);
      return authored === undefined ? undefined : { id: model.id, model: authored as SyncedModel };
    }
    return {
      id: model.id,
      model: buildRequestyModel(model, baseModel, context.existing(model.id)),
    };
  },
} satisfies SyncProvider<RequestyModel>;

export function buildRequestyModel(
  model: RequestyModel,
  baseModel: string,
  existing: ExistingModel | undefined,
): SyncedModel {
  const base = baseMetadata(baseModel);
  const built = buildOpenRouterModel(
    toOpenRouterModel(model, base),
    existing,
    baseModel,
  ) as SyncedBaseModel;
  // Name, description, and family are provider-agnostic facts of the base model.
  const { name: _name, description: _description, family: _family, ...factored } = built;
  const tiers = pricingTiers(model) ?? existing?.cost?.tiers;
  return tiers === undefined ? factored : { ...factored, cost: { ...factored.cost, tiers } };
}

function toOpenRouterModel(model: RequestyModel, base: BaseMetadata): OpenRouterModel {
  const context = model.context_window > 0 ? model.context_window : base.limit.context;
  const reasoning = model.supports_reasoning === true || base.reasoning === true;
  return {
    id: model.id,
    name: "",
    created: model.created,
    hugging_face_id: base.open_weights === true ? model.id : null,
    knowledge_cutoff: null,
    context_length: context,
    architecture: {
      input_modalities: base.modalities.input,
      output_modalities: base.modalities.output,
    },
    pricing: {
      prompt: String(model.input_price),
      completion: String(model.output_price),
      input_cache_read: chargedPerTokenPrice(model.cached_price),
      input_cache_write: chargedPerTokenPrice(model.caching_price),
    },
    top_provider: {
      context_length: context,
      max_completion_tokens: model.max_output_tokens > 0
        ? model.max_output_tokens
        : base.limit.output ?? null,
    },
    supported_parameters: [
      ...(base.temperature === false ? [] : ["temperature"]),
      ...(reasoning ? ["reasoning"] : []),
      ...(model.supports_tool_calling === true || base.tool_call === true ? ["tools"] : []),
      ...(model.supports_output_json_schema === true || base.structured_output === true
        ? ["structured_outputs"]
        : []),
    ],
    // Requesty translates a single `reasoning_effort` into each vendor's native
    // reasoning control: https://docs.requesty.ai/features/reasoning
    reasoning: reasoning ? { mandatory: false, supported_efforts: [...REASONING_EFFORTS] } : undefined,
  };
}

/** Context-length pricing bands. The first band is the flat `cost` of the model. */
function pricingTiers(model: RequestyModel): NonNullable<SyncedFullModel["cost"]>["tiers"] {
  const tiers = (model.pricing ?? [])
    .slice(1)
    .map((band) => ({
      tier: { type: "context" as const, size: band.prompt_tokens_threshold },
      input: pricePerMillion(band.input_price ?? model.input_price),
      output: pricePerMillion(band.output_price ?? model.output_price),
      cache_read: chargedPricePerMillion(band.cached_price),
      cache_write: chargedPricePerMillion(band.caching_price),
    }));
  return tiers.length > 0 ? tiers : undefined;
}

/** Requesty prices are USD per token; zero means the route does not charge for it. */
function chargedPerTokenPrice(price: number | undefined): string | undefined {
  return price === undefined || price <= 0 ? undefined : String(price);
}

function chargedPricePerMillion(price: number | undefined): number | undefined {
  return price === undefined || price <= 0 ? undefined : pricePerMillion(price);
}

function pricePerMillion(price: number): number {
  return Math.round(price * TOKENS_PER_MILLION * PRICE_DECIMALS) / PRICE_DECIMALS;
}

const metadataBySlug = new Map<string, string | undefined>();
const metadataByID = new Map<string, BaseMetadata>();

/** `vertex/claude-opus-4@us-east5` and `anthropic/claude-opus-4` are the same model. */
export function resolveRequestyBaseModel(modelID: string): string | undefined {
  if (metadataBySlug.size === 0) indexMetadata();
  const slug = modelID.split("/").at(-1)?.split(/[@:]/)[0];
  return slug === undefined ? undefined : metadataBySlug.get(slug.toLowerCase());
}

function indexMetadata() {
  for (const provider of readdirSync(MODELS_DIR)) {
    for (const file of readdirSync(path.join(MODELS_DIR, provider))) {
      if (!file.endsWith(".toml")) continue;
      const modelID = file.slice(0, -".toml".length);
      const slug = modelID.toLowerCase();
      // An ambiguous slug cannot be attributed to one lab from the route alone.
      metadataBySlug.set(slug, metadataBySlug.has(slug) ? undefined : `${provider}/${modelID}`);
    }
  }
}

function baseMetadata(modelID: string): BaseMetadata {
  let metadata = metadataByID.get(modelID);
  if (metadata === undefined) {
    const file = readFileSync(path.join(MODELS_DIR, `${modelID}.toml`), "utf8");
    metadata = BaseMetadata.parse(Bun.TOML.parse(file));
    metadataByID.set(modelID, metadata);
  }
  return metadata;
}
