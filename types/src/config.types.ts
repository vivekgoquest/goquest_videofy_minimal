import { z } from "zod";
import { cameraMovementsEnum } from "./constants.types";

const backgroundSchema = z.object({
  max_volume: z.number().positive(),
  min_volume: z.number().positive(),
  music_map: z.string().optional(),
  music: z
    .record(z.string(), z.union([z.string(), z.array(z.string())]))
    .optional(),
});

const graphicsSchema = z.object({
  item_types: z.array(
    z.union([
      z.literal("text"),
      z.literal("image"),
      z.literal("video"),
      z.literal("map"),
    ])
  ),
});

const manuscriptConfigSchema = z.object({
  split_article_into_chapters: z.boolean(),
  intro: z.string().optional(),
  outro: z.string().optional(),
  script_prompt: z.string(),
  placement_prompt: z.string(),
  describe_images_prompt: z.string(),
});

const llmProviderSchema = z.union([z.literal("openai"), z.literal("gemini")]);

const llmNodeSchema = z.object({
  provider: llmProviderSchema.optional(),
  model: z.string().optional(),
});

const llmSchema = z.object({
  default_provider: llmProviderSchema,
  nodes: z.object({
    script_generation: llmNodeSchema.optional(),
    image_description: llmNodeSchema.optional(),
    asset_placement: llmNodeSchema.optional(),
    image_prompt_builder: llmNodeSchema.optional(),
  }),
});

const imageGenerationProviderSchema = z.union([
  z.literal("openai"),
  z.literal("nanobanana"),
]);

const imageGenerationPromptsSchema = z.object({
  brief_prompt: z.string().optional(),
  openai_prompt_builder: z.string(),
  nanobanana_prompt_builder: z.string(),
});

const openAiImageGenerationSchema = z.object({
  model: z.string().optional(),
  size: z
    .union([
      z.literal("auto"),
      z.literal("1024x1024"),
      z.literal("1536x1024"),
      z.literal("1024x1536"),
    ])
    .optional(),
  quality: z
    .union([
      z.literal("auto"),
      z.literal("low"),
      z.literal("medium"),
      z.literal("high"),
    ])
    .optional(),
  background: z
    .union([z.literal("auto"), z.literal("transparent"), z.literal("opaque")])
    .optional(),
});

const nanobananaImageGenerationSchema = z.object({
  model: z.string().optional(),
  aspect_ratio: z
    .union([z.literal("1:1"), z.literal("9:16"), z.literal("16:9")])
    .optional(),
  thinking_budget: z
    .union([
      z.literal("none"),
      z.literal("low"),
      z.literal("medium"),
      z.literal("high"),
    ])
    .optional(),
});

const imageGenerationSchema = z.object({
  enabled: z.boolean(),
  provider: imageGenerationProviderSchema,
  prompt_builder_model: z.string().optional(),
  variants: z.number().int().min(1).max(4),
  prefer_generated: z.boolean(),
  prompts: imageGenerationPromptsSchema,
  openai: openAiImageGenerationSchema.optional(),
  nanobanana: nanobananaImageGenerationSchema.optional(),
});

const personSchema = z.object({
  voice: z.string(),
  model_id: z.string().optional(),
  modelId: z.string().optional(),
  instructions: z.string().optional(),
  stability: z.number().optional(),
  similarity_boost: z.number().optional(),
  style: z.number().optional(),
  use_speaker_boost: z.boolean().optional(),
});

const audioSchema = z.object({
  tts: z.literal("google"),
  background: backgroundSchema,
  sync_silence: z.number(),
  segment_pause: z.number(),
  segment_pause_silence: z.number(),
});

const peopleSchema = z.object({
  default: personSchema,
  intro: personSchema.optional(),
  outro: personSchema.optional(),
});

const styleSchema = z.object({
  captions: z
    .object({
      container: z.string().optional(),
      text: z.string().optional(),
      placements: z
        .object({
          top: z.string().optional(),
          middle: z.string().optional(),
          bottom: z.string().optional(),
        })
        .optional(),
    })
    .optional(),
  photoCredits: z
    .object({
      container: z.string().optional(),
      text: z.string().optional(),
    })
    .optional(),
  progress: z
    .object({
      active: z.string(),
      inactive: z.string(),
    })
    .optional(),
  map: z
    .object({
      marker: z.object({
        color: z.string(),
        scale: z.number(),
      }),
    })
    .optional(),
});

const stylesSchema = z.object({
  all: styleSchema.optional(),
  portrait: styleSchema.optional(),
  landscape: styleSchema.optional(),
});

export const colorsSchema = z.object({
  text: z.object({
    text: z.string(),
    background: z.string(),
  }),
  progress: z.object({
    active: z.object({
      background: z.string(),
      text: z.string(),
    }),
    inactive: z.object({
      background: z.string(),
      text: z.string(),
    }),
  }),
  map: z.object({
    marker: z.string(),
  }),
  fotoCredits: z.object({
    text: z.string(),
    icon: z.string(),
  }),
});

export const customCutsSchema = z.object({
  portrait: z.string(),
  landscape: z.string(),
  duration: z.number(),
  offset: z.number(),
});

export const reporterVideoSchema = z.object({
  // Renamed from videoSchema
  // This is for reporter videos
  portrait: z.string(),
  landscape: z.string(),
  duration: z.number(),
  selected: z.boolean().optional(),
});

const reporterVideosSchema = z.object({
  intro: z.record(z.string(), reporterVideoSchema).optional(),
  outro: z.record(z.string(), reporterVideoSchema).optional(),
});

const fontsSchema = z.record(z.string(), z.string());

export const playerSchema = z.object({
  assetBaseUrl: z.string().optional(),
  logo: z.string(),
  logoStyle: z.string().optional(),
  defaultCameraMovements: z.array(cameraMovementsEnum).optional(),
  fonts: fontsSchema.optional(),
  styles: stylesSchema.optional(),
  colors: colorsSchema.optional(),
  backgroundMusicVolume: z.number().optional(),
  backgroundMusic: z.string().optional(),
  moodMusic: z
    .record(z.string(), z.union([z.string(), z.array(z.string())]))
    .optional(),
  intro: customCutsSchema.optional(),
  wipe: customCutsSchema.optional(),
  outro: customCutsSchema.optional(),
  reporterVideos: reporterVideosSchema.optional(),
});

const streamSchema = z.object({
  enabled: z.boolean(),
  provider: z.string(),
  providerTitle: z.string(),
  titlePrefix: z.string(),
  verticalCategory: z.number(),
  horizontalCategory: z.number(),
});

const exportDefaultsSchema = z.object({
  exportType: z.string().optional(),
  logo: z.boolean().optional(),
  audio: z.boolean().optional(),
  voice: z.boolean().optional(),
  music: z.boolean().optional(),
});

const configSchema = z.object({
  name: z.string(),
  description: z.string(),
  id: z.string(),
  people: peopleSchema,
  manuscript: manuscriptConfigSchema,
  llm: llmSchema.optional(),
  image_generation: imageGenerationSchema.optional(),
  graphics: graphicsSchema,
  audio: audioSchema,
  player: playerSchema.optional(),
  stream: streamSchema.optional(),
  exportDefaults: exportDefaultsSchema.optional(),
  path: z.string().optional(),
  default_assets_base_url: z.string(),
});

export const appConfigSchema = configSchema.extend({
  player: playerSchema.optional(),
  stream: streamSchema.optional(),
  path: z.string().optional(),
});

export type Config = z.infer<typeof appConfigSchema>;
