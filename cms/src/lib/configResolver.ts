import { readdir, readFile } from "node:fs/promises";
import type { Config } from "@videofy/types";
import { buildDefaultConfig } from "@/lib/defaultConfig";
import {
  GenerationManifest,
  configOverridePath,
  configRoot,
  readJson,
} from "@/lib/projectFiles";

type ImageGenerationConfig = NonNullable<Config["image_generation"]>;
type LLMConfig = NonNullable<Config["llm"]>;
type LLMNodeKey = keyof LLMConfig["nodes"];
type LLMNodeConfig = NonNullable<LLMConfig["nodes"][LLMNodeKey]>;

function deepMerge(base: Record<string, unknown>, patch: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { ...base };
  for (const [k, v] of Object.entries(patch)) {
    const current = out[k];
    if (
      current &&
      typeof current === "object" &&
      !Array.isArray(current) &&
      v &&
      typeof v === "object" &&
      !Array.isArray(v)
    ) {
      out[k] = deepMerge(current as Record<string, unknown>, v as Record<string, unknown>);
    } else {
      out[k] = v;
    }
  }
  return out;
}

async function readObject(filePath: string): Promise<Record<string, unknown>> {
  try {
    const raw = await readFile(filePath, "utf-8");
    const parsed = JSON.parse(raw);
    return typeof parsed === "object" && parsed ? (parsed as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

async function readBrandObject(brandId: string): Promise<Record<string, unknown>> {
  const brandsDir = configRoot();
  const directPath = `${brandsDir}/${brandId}.json`;
  const direct = await readObject(directPath);
  if (Object.keys(direct).length > 0) {
    return direct;
  }

  try {
    const entries = await readdir(brandsDir, { withFileTypes: true });
    const jsonFiles = entries
      .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(".json"))
      .map((entry) => entry.name)
      .sort((a, b) => a.localeCompare(b));
    if (jsonFiles.length === 0) {
      return {};
    }
    const fallback =
      jsonFiles.find((fileName) => fileName.toLowerCase() === "default.json") || jsonFiles[0];
    return await readObject(`${brandsDir}/${fallback}`);
  } catch {
    return {};
  }
}

export async function resolveConfigForProject(
  projectId: string,
  manifest: GenerationManifest
): Promise<Config> {
  const brand = await readBrandObject(manifest.brandId);
  const override = await readJson<Record<string, unknown>>(configOverridePath(projectId), {});

  const merged = brand;

  const config = buildDefaultConfig(projectId);
  const prompts = (merged.prompts || {}) as Record<string, unknown>;
  const brandPeople = (brand.people || {}) as Record<string, unknown>;
  const people = brandPeople;
  const defaultPerson = ((people.default || {}) as Record<string, unknown>);
  const audio = (merged.audio || {}) as Record<string, unknown>;
  const options = (merged.options || {}) as Record<string, unknown>;

  config.manuscript.script_prompt =
    (prompts.scriptPrompt as string) || config.manuscript.script_prompt;
  config.manuscript.placement_prompt =
    (prompts.placementPrompt as string) || config.manuscript.placement_prompt;
  config.manuscript.describe_images_prompt =
    (prompts.describeImagesPrompt as string) || config.manuscript.describe_images_prompt;

  const imageGeneration = (merged.imageGeneration || {}) as Record<string, unknown>;
  const imageGenerationPrompts = (imageGeneration.prompts || {}) as Record<string, unknown>;
  const openaiImageGeneration = (imageGeneration.openai || {}) as Record<string, unknown>;
  const nanobananaImageGeneration = (imageGeneration.nanobanana || {}) as Record<string, unknown>;
  const llm = (merged.llm || {}) as Record<string, unknown>;
  const llmNodes = (llm.nodes || {}) as Record<string, unknown>;
  const openai = (merged.openai || {}) as Record<string, unknown>;
  const gemini = (merged.gemini || {}) as Record<string, unknown>;

  if (config.llm) {
    const defaultProvider =
      (llm.defaultProvider as LLMConfig["default_provider"]) ||
      (llm.default_provider as LLMConfig["default_provider"]);
    if (defaultProvider === "openai" || defaultProvider === "gemini") {
      config.llm.default_provider = defaultProvider;
    }

    const readNodeObject = (camelKey: string, snakeKey: string) => {
      const camelValue = llmNodes[camelKey];
      if (camelValue && typeof camelValue === "object" && !Array.isArray(camelValue)) {
        return camelValue as Record<string, unknown>;
      }
      const snakeValue = llmNodes[snakeKey];
      if (snakeValue && typeof snakeValue === "object" && !Array.isArray(snakeValue)) {
        return snakeValue as Record<string, unknown>;
      }
      return {};
    };

    const applyNode = (
      nodeKey: LLMNodeKey,
      rawNode: Record<string, unknown>,
      fallbackModels: {
        openai?: string;
        gemini?: string;
      }
    ) => {
      const currentNode = (config.llm?.nodes[nodeKey] || {}) as LLMNodeConfig;
      const nextNode: LLMNodeConfig = { ...currentNode };

      if (rawNode.provider === "openai" || rawNode.provider === "gemini") {
        nextNode.provider = rawNode.provider;
      }

      if (typeof rawNode.model === "string" && rawNode.model) {
        nextNode.model = rawNode.model;
      } else {
        const provider = nextNode.provider || config.llm?.default_provider || "openai";
        const fallbackModel =
          provider === "gemini" ? fallbackModels.gemini : fallbackModels.openai;
        if (typeof fallbackModel === "string" && fallbackModel) {
          nextNode.model = fallbackModel;
        }
      }

      config.llm!.nodes[nodeKey] = nextNode;
    };

    const legacyPromptBuilderModel =
      typeof imageGeneration.promptBuilderModel === "string"
        ? imageGeneration.promptBuilderModel
        : undefined;

    applyNode(
      "script_generation",
      readNodeObject("scriptGeneration", "script_generation"),
      {
        openai:
          typeof openai.manuscriptModel === "string" ? openai.manuscriptModel : "gpt-4o-mini",
        gemini:
          typeof gemini.manuscriptModel === "string"
            ? gemini.manuscriptModel
            : "gemini-2.5-flash",
      }
    );
    applyNode(
      "image_description",
      readNodeObject("imageDescription", "image_description"),
      {
        openai:
          typeof openai.mediaModel === "string"
            ? openai.mediaModel
            : typeof openai.manuscriptModel === "string"
              ? openai.manuscriptModel
              : "gpt-4o-mini",
        gemini:
          typeof gemini.mediaModel === "string"
            ? gemini.mediaModel
            : typeof gemini.manuscriptModel === "string"
              ? gemini.manuscriptModel
              : "gemini-2.5-flash",
      }
    );
    applyNode(
      "asset_placement",
      readNodeObject("assetPlacement", "asset_placement"),
      {
        openai:
          typeof openai.mediaModel === "string"
            ? openai.mediaModel
            : typeof openai.manuscriptModel === "string"
              ? openai.manuscriptModel
              : "gpt-4o-mini",
        gemini:
          typeof gemini.mediaModel === "string"
            ? gemini.mediaModel
            : typeof gemini.manuscriptModel === "string"
              ? gemini.manuscriptModel
              : "gemini-2.5-flash",
      }
    );
    applyNode(
      "image_prompt_builder",
      readNodeObject("imagePromptBuilder", "image_prompt_builder"),
      {
        openai: legacyPromptBuilderModel || "gpt-5-mini",
        gemini:
          (typeof gemini.promptBuilderModel === "string" && gemini.promptBuilderModel) ||
          legacyPromptBuilderModel ||
          "gemini-2.5-flash",
      }
    );
  }

  if (typeof imageGeneration.enabled === "boolean" && config.image_generation) {
    config.image_generation.enabled = imageGeneration.enabled;
  }
  if (typeof imageGeneration.provider === "string" && config.image_generation) {
    config.image_generation.provider =
      imageGeneration.provider as ImageGenerationConfig["provider"];
  }
  if (typeof imageGeneration.promptBuilderModel === "string" && config.image_generation) {
    config.image_generation.prompt_builder_model = imageGeneration.promptBuilderModel;
  }
  if (typeof imageGeneration.variants === "number" && config.image_generation) {
    config.image_generation.variants = imageGeneration.variants;
  }
  if (typeof imageGeneration.preferGenerated === "boolean" && config.image_generation) {
    config.image_generation.prefer_generated = imageGeneration.preferGenerated;
  }
  if (typeof imageGenerationPrompts.briefPrompt === "string" && config.image_generation) {
    config.image_generation.prompts.brief_prompt = imageGenerationPrompts.briefPrompt;
  }
  if (
    typeof imageGenerationPrompts.openaiPromptBuilder === "string" &&
    config.image_generation
  ) {
    config.image_generation.prompts.openai_prompt_builder =
      imageGenerationPrompts.openaiPromptBuilder;
  }
  if (
    typeof imageGenerationPrompts.nanobananaPromptBuilder === "string" &&
    config.image_generation
  ) {
    config.image_generation.prompts.nanobanana_prompt_builder =
      imageGenerationPrompts.nanobananaPromptBuilder;
  }

  if (config.image_generation) {
    config.image_generation.openai = {
      ...config.image_generation.openai,
    ...(typeof openaiImageGeneration.model === "string"
      ? { model: openaiImageGeneration.model }
      : {}),
    ...(typeof openaiImageGeneration.size === "string"
      ? { size: openaiImageGeneration.size as NonNullable<ImageGenerationConfig["openai"]>["size"] }
      : {}),
    ...(typeof openaiImageGeneration.quality === "string"
      ? {
          quality:
            openaiImageGeneration.quality as NonNullable<ImageGenerationConfig["openai"]>["quality"],
        }
      : {}),
    ...(typeof openaiImageGeneration.background === "string"
      ? {
          background:
            openaiImageGeneration.background as NonNullable<ImageGenerationConfig["openai"]>["background"],
        }
      : {}),
    };

    config.image_generation.nanobanana = {
      ...config.image_generation.nanobanana,
    ...(typeof nanobananaImageGeneration.model === "string"
      ? { model: nanobananaImageGeneration.model }
      : {}),
    ...(typeof nanobananaImageGeneration.aspectRatio === "string"
      ? {
          aspect_ratio:
            nanobananaImageGeneration.aspectRatio as NonNullable<ImageGenerationConfig["nanobanana"]>["aspect_ratio"],
        }
      : {}),
    ...(typeof nanobananaImageGeneration.thinkingBudget === "string"
      ? {
          thinking_budget:
            nanobananaImageGeneration.thinkingBudget as NonNullable<ImageGenerationConfig["nanobanana"]>["thinking_budget"],
        }
      : {}),
    };
  }

  if (brandPeople && typeof brandPeople === "object") {
    config.people = deepMerge(
      (config.people || {}) as Record<string, unknown>,
      brandPeople as Record<string, unknown>
    ) as Config["people"];
  }

  if (typeof defaultPerson.voice === "string" && defaultPerson.voice) {
    config.people.default.voice = defaultPerson.voice;
  }
  if (typeof defaultPerson.model_id === "string" && defaultPerson.model_id) {
    config.people.default.model_id = defaultPerson.model_id;
  } else if (typeof defaultPerson.modelId === "string" && defaultPerson.modelId) {
    config.people.default.model_id = defaultPerson.modelId;
  }
  if (typeof defaultPerson.instructions === "string" && defaultPerson.instructions) {
    config.people.default.instructions = defaultPerson.instructions;
  }

  if (typeof defaultPerson.stability === "number") {
    config.people.default.stability = defaultPerson.stability;
  }
  if (typeof defaultPerson.similarity_boost === "number") {
    config.people.default.similarity_boost = defaultPerson.similarity_boost;
  }
  if (typeof defaultPerson.style === "number") {
    config.people.default.style = defaultPerson.style;
  }
  if (typeof defaultPerson.use_speaker_boost === "boolean") {
    config.people.default.use_speaker_boost = defaultPerson.use_speaker_boost;
  }

  if (typeof options.segmentPauseSeconds === "number") {
    config.audio.segment_pause = options.segmentPauseSeconds;
  }
  if (audio.tts === "google") {
    config.audio.tts = "google";
  }

  if (merged.player && typeof merged.player === "object") {
    config.player = deepMerge(
      (config.player || {}) as Record<string, unknown>,
      merged.player as Record<string, unknown>
    ) as Config["player"];
  }

  config.player = {
    ...(config.player || {}),
    logo: config.player?.logo || "/assets/logo.svg",
    assetBaseUrl:
      process.env.NEXT_PUBLIC_CMS_BASE_URL || "http://127.0.0.1:3000",
  };

  if (merged.exportDefaults && typeof merged.exportDefaults === "object") {
    config.exportDefaults = deepMerge(
      (config.exportDefaults || {}) as Record<string, unknown>,
      merged.exportDefaults as Record<string, unknown>
    ) as Config["exportDefaults"];
  }

  const fileBase = process.env.MINIMAL_FILE_BASE_URL || "http://127.0.0.1:8001";
  config.default_assets_base_url = `${fileBase}/projects/${projectId}/files/input`;

  if (override && Object.keys(override).length > 0) {
    return deepMerge(config as unknown as Record<string, unknown>, override) as unknown as Config;
  }

  return config;
}
