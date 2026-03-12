"use client";
import { useEffect, useMemo, useRef } from "react";
import { useReactive } from "ahooks";
import { generateManuscript } from "@/utils/generateManuscript";
import { useGlobalState } from "@/state/globalState";
import { Config } from "@videofy/types";
import Cookies from "universal-cookie";
import {
  Alert,
  App,
  Button,
  Card,
  Flex,
  Form,
  Input,
  Select,
  Spin,
  theme,
  Typography,
} from "antd";
import { useRouter } from "next/navigation";
import {
  getBrands,
  getConfigs,
  getProjects,
  runFetcherPlugin,
  setProjectBrand,
  useBrands,
  useFetchers,
} from "@/api";
import { LoadingOutlined } from "@ant-design/icons";

const { Title, Paragraph, Text } = Typography;
const cookies = new Cookies();

type LLMProvider = "openai" | "gemini";
type LLMNodeKey =
  | "script_generation"
  | "image_description"
  | "asset_placement"
  | "image_prompt_builder";
type ImageProvider = "openai" | "nanobanana";
type ModelSelection = "__provider_default__" | "__custom__" | string;

type FormType = {
  fetcherId: string;
  brandId: string;
  prompt: string;
  defaultLlmProvider: "brand-default" | LLMProvider;
  llmNodeOverrides: Record<LLMNodeKey, "default-pipeline" | LLMProvider>;
  llmNodeModelSelections: Record<LLMNodeKey, ModelSelection>;
  llmNodeCustomModels: Record<LLMNodeKey, string>;
  imageProvider: "brand-default" | "disabled" | "openai" | "nanobanana";
  imageModelSelection: "__provider_default__" | "__custom__" | string;
  imageCustomModel: string;
  inputs: Record<string, string>;
};

const PROVIDER_DEFAULT = "__provider_default__";
const CUSTOM_MODEL = "__custom__";

const llmNodeOptions: Array<{
  key: LLMNodeKey;
  label: string;
  description: string;
}> = [
  {
    key: "script_generation",
    label: "Script generation",
    description: "Turns the source article into short narration lines.",
  },
  {
    key: "image_description",
    label: "Image description",
    description: "Explains what each imported visual contains.",
  },
  {
    key: "asset_placement",
    label: "Asset placement",
    description: "Matches the best visual to each script beat.",
  },
  {
    key: "image_prompt_builder",
    label: "Image prompt builder",
    description: "Builds the final prompt structure for AI image generation.",
  },
];

const llmModelPresets: Record<LLMNodeKey, Record<LLMProvider, string[]>> = {
  script_generation: {
    openai: ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    gemini: ["gemini-2.5-flash", "gemini-2.5-pro"],
  },
  image_description: {
    openai: ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"],
    gemini: ["gemini-2.5-flash", "gemini-2.5-pro"],
  },
  asset_placement: {
    openai: ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"],
    gemini: ["gemini-2.5-flash", "gemini-2.5-pro"],
  },
  image_prompt_builder: {
    openai: ["gpt-5-mini", "gpt-4o", "gpt-4.1-mini"],
    gemini: ["gemini-2.5-flash", "gemini-2.5-pro"],
  },
};

const imageModelPresets: Record<ImageProvider, string[]> = {
  openai: ["gpt-5-mini"],
  nanobanana: ["gemini-2.5-flash-image-preview"],
};

function resolveEffectiveLlmProvider(
  defaultProvider: FormType["defaultLlmProvider"],
  nodeProvider: FormType["llmNodeOverrides"][LLMNodeKey]
): LLMProvider | null {
  if (nodeProvider !== "default-pipeline") {
    return nodeProvider;
  }
  if (defaultProvider !== "brand-default") {
    return defaultProvider;
  }
  return null;
}

function normalizeModelOverride(
  selection: string | undefined,
  customValue: string | undefined
): string | undefined {
  if (!selection || selection === PROVIDER_DEFAULT) {
    return undefined;
  }
  if (selection === CUSTOM_MODEL) {
    const trimmed = (customValue || "").trim();
    return trimmed || undefined;
  }
  return selection;
}

function buildCustomizedLLMConfig(
  config: Config,
  values: FormType
): Config["llm"] {
  if (!config.llm) {
    return config.llm;
  }

  const nextDefaultProvider =
    values.defaultLlmProvider === "brand-default"
      ? config.llm.default_provider
      : values.defaultLlmProvider;
  const keepBrandPipeline = values.defaultLlmProvider === "brand-default";

  return {
    ...config.llm,
    default_provider: nextDefaultProvider,
    nodes: Object.fromEntries(
      llmNodeOptions.map(({ key }) => {
        const currentNode = config.llm?.nodes[key] || {};
        const selectedOverride = values.llmNodeOverrides?.[key] || "default-pipeline";
        const modelOverride = normalizeModelOverride(
          values.llmNodeModelSelections?.[key],
          values.llmNodeCustomModels?.[key]
        );

        if (keepBrandPipeline && selectedOverride === "default-pipeline") {
          return [key, currentNode];
        }

        return [
          key,
          {
            ...(selectedOverride === "default-pipeline"
              ? { provider: undefined }
              : { provider: selectedOverride }),
            ...(modelOverride ? { model: modelOverride } : { model: undefined }),
          },
        ];
      })
    ) as NonNullable<Config["llm"]>["nodes"],
  };
}

function buildCustomizedImageGenerationConfig(
  config: Config,
  values: FormType
): Config["image_generation"] {
  if (!config.image_generation) {
    return config.image_generation;
  }

  if (values.imageProvider === "brand-default") {
    return config.image_generation;
  }

  const explicitProvider =
    values.imageProvider !== "disabled" ? values.imageProvider : null;
  const nextConfig: NonNullable<Config["image_generation"]> = {
    ...config.image_generation,
    enabled: values.imageProvider === "disabled" ? false : config.image_generation.enabled,
    provider:
      explicitProvider || config.image_generation.provider,
  };
  const modelOverride = normalizeModelOverride(
    values.imageModelSelection,
    values.imageCustomModel
  );

  nextConfig.enabled = values.imageProvider !== "disabled";
  if (explicitProvider) {
    nextConfig.provider = explicitProvider;
  }

  if (modelOverride && nextConfig.provider === "openai") {
    nextConfig.openai = {
      ...nextConfig.openai,
      model: modelOverride,
    };
  }
  if (modelOverride && nextConfig.provider === "nanobanana") {
    nextConfig.nanobanana = {
      ...nextConfig.nanobanana,
      model: modelOverride,
    };
  }

  return nextConfig;
}

const StartPage = () => {
  const { data: fetchers, isLoading: loadingFetchers } = useFetchers();
  const { data: brands, isLoading: loadingBrands } = useBrands();
  const hasFetchers = Boolean(fetchers && fetchers.length > 0);
  const hasBrands = Boolean(brands && brands.length > 0);

  const state = useReactive({
    loading: false,
    loadingMessage: "Generating video...",
  });
  const { notification } = App.useApp();
  const {
    setConfig,
    setCustomPrompt,
    setTabs,
    setCurrentTabIndex,
    setGenerationId,
    setSelectedProject,
  } = useGlobalState();
  const { token } = theme.useToken();

  const [form] = Form.useForm<FormType>();
  const selectedFetcherId = Form.useWatch("fetcherId", form);
  const selectedBrandId = Form.useWatch("brandId", form);
  const selectedDefaultLlmProvider = Form.useWatch("defaultLlmProvider", form);
  const selectedNodeOverrides = Form.useWatch("llmNodeOverrides", form);
  const selectedNodeModelSelections = Form.useWatch("llmNodeModelSelections", form);
  const selectedImageProvider = Form.useWatch("imageProvider", form);
  const selectedImageModelSelection = Form.useWatch("imageModelSelection", form);
  const selectedFetcher = useMemo(
    () => fetchers?.find((fetcher) => fetcher.id === selectedFetcherId),
    [fetchers, selectedFetcherId]
  );
  const selectedFetcherFields = useMemo(
    () =>
      (selectedFetcher?.fields || []).filter((field) => field.name !== "project_id"),
    [selectedFetcher]
  );
  const selectedBrand = useMemo(
    () => brands?.find((brand) => brand.id === selectedBrandId),
    [brands, selectedBrandId]
  );
  const lastSyncedBrandId = useRef<string | undefined>(undefined);

  const router = useRouter();

  useEffect(() => {
    if (!fetchers || fetchers.length === 0) {
      return;
    }
    const currentFetcherId = form.getFieldValue("fetcherId");
    if (currentFetcherId) {
      return;
    }
    form.setFieldsValue({
      fetcherId: fetchers[0].id,
      inputs: {},
    });
  }, [fetchers, form]);

  useEffect(() => {
    if (!brands || brands.length === 0) {
      return;
    }
    const currentBrandId = form.getFieldValue("brandId");
    if (currentBrandId) {
      return;
    }
    const initialBrand = brands[0];
    form.setFieldsValue({
      brandId: initialBrand.id,
      defaultLlmProvider: "brand-default",
      llmNodeOverrides: {
        script_generation: "default-pipeline",
        image_description: "default-pipeline",
        asset_placement: "default-pipeline",
        image_prompt_builder: "default-pipeline",
      },
      llmNodeModelSelections: {
        script_generation: PROVIDER_DEFAULT,
        image_description: PROVIDER_DEFAULT,
        asset_placement: PROVIDER_DEFAULT,
        image_prompt_builder: PROVIDER_DEFAULT,
      },
      llmNodeCustomModels: {
        script_generation: "",
        image_description: "",
        asset_placement: "",
        image_prompt_builder: "",
      },
      imageProvider: "brand-default",
      imageModelSelection: PROVIDER_DEFAULT,
      imageCustomModel: "",
      prompt: initialBrand.scriptPrompt || "",
    });
  }, [brands, form]);

  useEffect(() => {
    if (!brands || brands.length === 0 || !selectedBrandId) {
      return;
    }
    if (lastSyncedBrandId.current === selectedBrandId) {
      return;
    }
    const selectedBrand = brands.find((brand) => brand.id === selectedBrandId);
    if (!selectedBrand) {
      return;
    }
    const nextPrompt = selectedBrand.scriptPrompt || "";
    const currentPrompt = form.getFieldValue("prompt") || "";
    if (nextPrompt !== currentPrompt) {
      form.setFields([{ name: "prompt", value: nextPrompt }]);
    }
    lastSyncedBrandId.current = selectedBrandId;
  }, [brands, form, selectedBrandId]);

  useEffect(() => {
    if (!selectedFetcherId) {
      return;
    }
    form.setFieldValue("inputs", {});
  }, [selectedFetcherId, form]);

  const loadManuscript = async (values: FormType) => {
    const { prompt, fetcherId, brandId } = values;
    const customPrompt = (prompt || "").trim();
    const selected = fetchers?.find((fetcher) => fetcher.id === fetcherId);
    if (!selected) {
      notification.error({ title: "Fetcher not found." });
      return;
    }
    const brand =
      brands?.find((item) => item.id === brandId) ||
      (await getBrands()).find((item) => item.id === brandId);
    if (!brand) {
      notification.error({ title: "Brand not found." });
      return;
    }
    state.loading = true;
    state.loadingMessage = "Fetching article...";
    try {
      const fetchResult = await runFetcherPlugin({
        fetcherId,
        inputs: values.inputs || {},
      });

      state.loadingMessage = "Applying brand settings...";
      await setProjectBrand(fetchResult.projectId, brand.id);

      state.loadingMessage = "Loading project configuration...";
      const [projects, configs] = await Promise.all([getProjects(), getConfigs()]);
      const project = projects.find((p) => p.id === fetchResult.projectId) || {
        id: fetchResult.projectId,
        name: fetchResult.projectId,
      };
      setSelectedProject(project);

      const configRow = configs.find((c) => c.projectId === fetchResult.projectId);
      const config = configRow?.config;
      if (!configRow || !config) {
        throw new Error(`Config not found for project '${fetchResult.projectId}'`);
      }

      const customizedConfig: Config = {
        ...config,
        manuscript: {
          ...config.manuscript,
          script_prompt: customPrompt || config.manuscript.script_prompt,
        },
        llm: buildCustomizedLLMConfig(config, values),
        image_generation: buildCustomizedImageGenerationConfig(config, values),
      };

      setConfig({ ...configRow, config: customizedConfig });

      state.loadingMessage = "Generating manuscript...";
      const manuscript = await generateManuscript(fetchResult.projectId, customizedConfig);

      if (!manuscript) {
        throw new Error("Backend did not return a manuscript");
      }

      const cleanedManuscript = {
        ...manuscript,
        meta: {
          ...manuscript.meta,
          articleUrl: fetchResult.projectId,
          uniqueId: crypto.randomUUID(),
        },
      };

      const tabsData = [
        {
          articleUrl: fetchResult.projectId,
          projectId: fetchResult.projectId,
          manuscript: cleanedManuscript,
        },
      ];
      setTabs(tabsData);
      setCustomPrompt(customPrompt);
      setCurrentTabIndex(0);

      const response = await fetch("/api/generations", {
        method: "POST",
        body: JSON.stringify({
          projectId: fetchResult.projectId,
          brandId: brand.id,
          config: customizedConfig,
          project,
          data: tabsData,
        }),
      });
      if (!response.ok) {
        throw new Error("Failed to create generation");
      }
      const { id: generationId } = await response.json();

      setGenerationId(generationId);
      cookies.set("projectId", fetchResult.projectId);

      router.push(`/${encodeURIComponent(generationId)}`);
    } catch (error) {
      if (error instanceof Error) {
        notification.error({ title: error.message, duration: 0 });
      } else {
        notification.error({ title: "Failed to fetch article", duration: 0 });
      }
    } finally {
      state.loading = false;
      state.loadingMessage = "Generating video...";
    }
  };

  if (loadingFetchers || loadingBrands) {
    return (
      <Spin description="Loading fetchers and brands..." fullscreen delay={500} />
    );
  }

  return (
    <Flex vertical gap="middle" align="center" className="mt-4 mb-4">
      <Title style={{ fontSize: 60, marginBottom: 10, marginTop: 40 }}>
        Videofy
      </Title>
      <Paragraph style={{ marginBottom: 36 }}>
        Select a fetcher, provide inputs, and generate a video.
      </Paragraph>
      <Card style={{ width: "100%", maxWidth: "80ch" }}>
        {!hasFetchers && (
          <Alert
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
            title="No fetchers found"
            description="Add fetchers under minimal/fetchers/<fetcherId>/fetcher.json, then refresh."
          />
        )}
        {!hasBrands && (
          <Alert
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
            title="No brands found"
            description="Add brand json files under minimal/brands/, then refresh."
          />
        )}
        <Form form={form} onFinish={loadManuscript} layout="vertical">
          <Form.Item name="fetcherId" label="Fetcher" rules={[{ required: true }]}>
            <Select
              showSearch
              filterOption={(input, option) =>
                (option?.label ?? "")
                  .toLowerCase()
                  .includes(input.toLowerCase())
              }
              disabled={!hasFetchers}
              options={fetchers?.map((fetcher) => ({
                value: fetcher.id,
                label: fetcher.title,
              }))}
            />
          </Form.Item>
          {selectedFetcher && selectedFetcher.description && (
            <Paragraph type="secondary" style={{ marginTop: -8 }}>
              {selectedFetcher.description}
            </Paragraph>
          )}
          {selectedFetcherFields.map((field) => (
            <Form.Item
              key={field.name}
              name={["inputs", field.name]}
              label={field.label}
              preserve={false}
              rules={
                field.required
                  ? [{ required: true, message: `${field.label} is required` }]
                  : undefined
              }
            >
              <Input placeholder={field.placeholder} />
            </Form.Item>
          ))}
          <Form.Item name="brandId" label="Brand" rules={[{ required: true }]}>
            <Select
              showSearch
              filterOption={(input, option) =>
                (option?.label ?? "")
                  .toLowerCase()
                  .includes(input.toLowerCase())
              }
              disabled={!hasBrands}
              options={brands?.map((brand) => ({
                value: brand.id,
                label: brand.brandName,
              }))}
            />
          </Form.Item>
          <Form.Item noStyle>
            <Form.Item
              label="Custom prompt"
              name="prompt"
              style={{ marginBottom: 0 }}
            >
              <Input.TextArea rows={10} />
            </Form.Item>
          </Form.Item>
          <Card
            size="small"
            title="LLM pipeline"
            style={{ marginTop: 16, borderRadius: 16 }}
          >
            <Flex vertical gap={12}>
              <Alert
                type="info"
                showIcon
                title="Set one default provider for the text pipeline, then override only the nodes that need a different provider or model."
              />
              <Form.Item
                label="Default pipeline provider"
                name="defaultLlmProvider"
                initialValue="brand-default"
                extra="Choose OpenAI or Gemini for all text-reasoning stages. Leave this on Brand default to keep the selected brand's behavior."
              >
                <Select
                  options={[
                    { value: "brand-default", label: "Brand default" },
                    { value: "openai", label: "OpenAI" },
                    { value: "gemini", label: "Gemini" },
                  ]}
                />
              </Form.Item>
              {llmNodeOptions.map((node) => {
                const selectedNodeProvider =
                  selectedNodeOverrides?.[node.key] || "default-pipeline";
                const effectiveProvider = resolveEffectiveLlmProvider(
                  selectedDefaultLlmProvider || "brand-default",
                  selectedNodeProvider
                );
                const modelOptions = effectiveProvider
                  ? [
                      { value: PROVIDER_DEFAULT, label: "Use provider default" },
                      ...Array.from(
                        new Set(llmModelPresets[node.key][effectiveProvider])
                      ).map((model) => ({
                        value: model,
                        label: model,
                      })),
                      { value: CUSTOM_MODEL, label: "Custom model ID" },
                    ]
                  : [
                      {
                        value: PROVIDER_DEFAULT,
                        label: "Choose a provider first",
                      },
                    ];
                const selectedModel =
                  selectedNodeModelSelections?.[node.key] || PROVIDER_DEFAULT;

                return (
                  <Card
                    key={node.key}
                    size="small"
                    style={{
                      borderRadius: 14,
                      background: token.colorBgElevated,
                      borderColor: token.colorBorderSecondary,
                    }}
                  >
                    <Flex vertical gap={8}>
                      <Flex vertical gap={2}>
                        <Text strong>{node.label}</Text>
                        <Text type="secondary">{node.description}</Text>
                      </Flex>
                      <Form.Item
                        label="Provider override"
                        name={["llmNodeOverrides", node.key]}
                        initialValue="default-pipeline"
                        style={{ marginBottom: 8 }}
                        extra={
                          effectiveProvider
                            ? `Effective provider: ${effectiveProvider}`
                            : "This node will follow the selected brand until you pick a provider override."
                        }
                      >
                        <Select
                          options={[
                            {
                              value: "default-pipeline",
                              label: "Use default pipeline",
                            },
                            { value: "openai", label: "OpenAI" },
                            { value: "gemini", label: "Gemini" },
                          ]}
                        />
                      </Form.Item>
                      <Form.Item
                        label="Model override"
                        name={["llmNodeModelSelections", node.key]}
                        initialValue={PROVIDER_DEFAULT}
                        style={{ marginBottom: 8 }}
                        extra={
                          effectiveProvider
                            ? "Leave this on provider default unless you want to pin a specific model for this node."
                            : "Model presets unlock once the provider is known."
                        }
                      >
                        <Select
                          disabled={!effectiveProvider}
                          options={modelOptions}
                        />
                      </Form.Item>
                      {selectedModel === CUSTOM_MODEL ? (
                        <Form.Item
                          label="Custom model ID"
                          name={["llmNodeCustomModels", node.key]}
                          style={{ marginBottom: 0 }}
                        >
                          <Input
                            placeholder={
                              effectiveProvider === "gemini"
                                ? "e.g. gemini-2.5-pro"
                                : "e.g. gpt-4.1-mini"
                            }
                          />
                        </Form.Item>
                      ) : null}
                    </Flex>
                  </Card>
                );
              })}
            </Flex>
          </Card>
          <Card
            size="small"
            title="AI image generation"
            style={{ marginTop: 16, borderRadius: 16 }}
          >
            <Flex vertical gap={12}>
              <Form.Item
                label="AI image provider"
                name="imageProvider"
                initialValue="brand-default"
                extra="Use the brand's image settings, disable AI images, or force a specific image provider for this run."
              >
                <Select
                  options={[
                    { value: "brand-default", label: "Brand default" },
                    { value: "disabled", label: "Disable AI image generation" },
                    { value: "openai", label: "OpenAI" },
                    { value: "nanobanana", label: "Nanobanana" },
                  ]}
                />
              </Form.Item>
              {selectedImageProvider &&
              selectedImageProvider !== "brand-default" &&
              selectedImageProvider !== "disabled" ? (
                <>
                  <Form.Item
                    label="Image model"
                    name="imageModelSelection"
                    initialValue={PROVIDER_DEFAULT}
                    extra="Leave this on provider default unless you want to pin a specific image model."
                  >
                    <Select
                      options={[
                        { value: PROVIDER_DEFAULT, label: "Use provider default" },
                        ...Array.from(
                          new Set(imageModelPresets[selectedImageProvider])
                        ).map((model) => ({
                          value: model,
                          label: model,
                        })),
                        { value: CUSTOM_MODEL, label: "Custom model ID" },
                      ]}
                    />
                  </Form.Item>
                  {selectedImageModelSelection === CUSTOM_MODEL ? (
                    <Form.Item
                      label="Custom image model ID"
                      name="imageCustomModel"
                      style={{ marginBottom: 0 }}
                    >
                      <Input
                        placeholder={
                          selectedImageProvider === "nanobanana"
                            ? "e.g. gemini-2.5-flash-image-preview"
                            : "e.g. gpt-5-mini"
                        }
                      />
                    </Form.Item>
                  ) : null}
                </>
              ) : null}
            </Flex>
          </Card>
          {selectedBrand ? (
            <Paragraph type="secondary" style={{ marginTop: 8 }}>
              Prompt source: {selectedBrand.brandName}
            </Paragraph>
          ) : null}
          <Form.Item style={{ marginTop: 16 }}>
            {state.loading ? (
              <Button
                type="primary"
                size="large"
                icon={<LoadingOutlined spin />}
                disabled
              >
                {state.loadingMessage}
              </Button>
            ) : (
              <Button
                htmlType="submit"
                type="primary"
                size="large"
                disabled={!hasFetchers || !hasBrands}
              >
                Generate video
              </Button>
            )}
          </Form.Item>
        </Form>
      </Card>
    </Flex>
  );
};

export default StartPage;
