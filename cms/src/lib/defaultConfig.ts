import type { Config } from "@videofy/types";

export const buildDefaultConfig = (projectId: string): Config => ({
  name: "Minimal Default",
  description: "Local default config for minimal mode",
  id: "minimal-default",
  people: {
    default: {
      voice: "Kore",
      model_id: "gemini-2.5-pro-preview-tts",
      instructions:
        "Deliver a thoughtful, clear narrative explainer with steady pacing.",
    },
  },
  manuscript: {
    split_article_into_chapters: false,
    script_prompt:
      "Summarize the article in short factual lines suitable for voice-over.",
    placement_prompt:
      "Match the best available media for each script line. Prioritize videos.",
    describe_images_prompt:
      "Describe the image content briefly and factually for short-form news video.",
  },
  llm: {
    default_provider: "openai",
    nodes: {
      script_generation: {
        provider: "openai",
        model: "gpt-4o-mini",
      },
      image_description: {
        provider: "openai",
        model: "gpt-4o-mini",
      },
      asset_placement: {
        provider: "openai",
        model: "gpt-4o-mini",
      },
      image_prompt_builder: {
        provider: "openai",
        model: "gpt-5-mini",
      },
    },
  },
  image_generation: {
    enabled: false,
    provider: "openai",
    prompt_builder_model: "",
    variants: 1,
    prefer_generated: true,
    prompts: {
      brief_prompt:
        "Create a single still image for a narrative explainer video segment. The image should feel intentional, editorial, and cinematic rather than like generic stock art.",
      openai_prompt_builder:
        "You create XML-style render prompts for OpenAI image generation. Convert the provided story context into a concrete still-image prompt with these sections: scene, setting, composition, lighting, color_palette, style, negative_constraints. Keep every field visual and physically observable. Prefer editorial illustration or stylized realism over fake photojournalism. Return concise, specific details that can be rendered in one still image.",
      nanobanana_prompt_builder:
        "You write production-ready prompts for Nano Banana image generation. Convert the provided story context into a natural-language prompt that clearly states the subject, environment, composition, lighting, palette, and rendering style. Keep it cinematic, emotionally direct, and suitable for a single still frame in a vertical explainer video. Avoid XML markup and avoid documentary claims that the image is a real photograph.",
    },
    openai: {
      model: "gpt-5-mini",
      size: "1024x1536",
      quality: "high",
      background: "opaque",
    },
    nanobanana: {
      model: "gemini-2.5-flash-image-preview",
      aspect_ratio: "9:16",
      thinking_budget: "low",
    },
  },
  graphics: {
    item_types: ["text", "image", "video", "map"],
  },
  audio: {
    tts: "google",
    background: {
      max_volume: 1,
      min_volume: 0.2,
      music: {},
    },
    sync_silence: 0.5,
    segment_pause: 0.4,
    segment_pause_silence: 0.4,
  },
  player: {
    logo: "/assets/logo.svg",
    assetBaseUrl:
      process.env.NEXT_PUBLIC_CMS_BASE_URL || "http://127.0.0.1:3000",
  },
  default_assets_base_url: `/projects/${projectId}/files/input`,
  exportDefaults: {
    exportType: "Vertical",
    logo: true,
    audio: true,
    voice: true,
    music: true,
  },
});
