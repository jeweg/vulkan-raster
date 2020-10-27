#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include "vulkan/vulkan.hpp"

#include "device.hpp"
#include "window.hpp"
#include "swap_chain.hpp"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define GLM_SWIZZLE_XYZW
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <iostream>
#include <chrono>


VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE


VKAPI_ATTR VkBool32 VKAPI_CALL debug_utils_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_types,
    VkDebugUtilsMessengerCallbackDataEXT const *pCallbackData,
    void *)
{
    std::cerr << "[" << vk::to_string(static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(message_severity)) << "]"
              << "(" << vk::to_string(static_cast<vk::DebugUtilsMessageTypeFlagBitsEXT>(message_types)) << ") "
              << pCallbackData->pMessage << "\n";
    if (message_severity
        & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)) {
        debug_break();
    }
    return VK_TRUE;
}


void glfwErrorCallback(int error, const char *description)
{
    std::cerr << "GLFW error " << std::to_string(error) << ": " << description << "\n";
}


// Collect out into unified vertices so we have a single vertex index into
// the (non-interleaved) attribute arrays.
struct IndexKey
{
    int vi, ni, ti;
    explicit IndexKey(int vi, int ni = 0, int ti = 0) : vi(vi), ni(vi), ti(ti) {}
    friend bool operator==(const IndexKey &a, const IndexKey &b)
    {
        return a.vi == b.vi && a.ni == b.ni && a.ti == b.ti;
    }
};
namespace std {
template <>
struct hash<IndexKey>
{
    std::size_t operator()(IndexKey const &k) const noexcept { return k.vi ^ (k.ni << 2) ^ (k.ti << 4); }
};
}


class Staging
{
public:
    explicit Staging(Device &device, size_t size_in_mib = 10) : _device(device), _staging_buffer_size_mib(size_in_mib)
    {}

    void copy(void *src_ptr, size_t src_bytes, VkBuffer dst_buffer, vk::DeviceSize dst_offset = 0)
    {
        const size_t batch_size = _staging_buffer_size_mib * 1024u * 1024u;
        if (!_staging_buffer.buffer) {
            _staging_buffer = std::move(VmaBuffer(
                _device.get_vma_allocator(),
                VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
                true,
                vk::BufferCreateInfo{}
                    .setUsage(vk::BufferUsageFlagBits::eTransferSrc)
                    .setSharingMode(vk::SharingMode::eExclusive)
                    .setSize(batch_size)));
        }
        const int batch_count = static_cast<int>(std::ceil(src_bytes / static_cast<float>(batch_size)));
        for (int i = 0; i < batch_count; ++i) {
            char *this_batch_src_ptr = static_cast<char *>(src_ptr) + i * batch_size;
            size_t this_batch_size = batch_size;
            if (i == batch_count - 1) { this_batch_size = src_bytes - i * batch_size; }

            std::memcpy(_staging_buffer.mapped_data(), this_batch_src_ptr, this_batch_size);

            _device.run_commands(Device::Queue::AsyncTransfer, [&](vk::CommandBuffer cmdbuf) {
                auto copy =
                    vk::BufferCopy{}.setSrcOffset(0).setDstOffset(dst_offset + i * batch_size).setSize(this_batch_size);
                cmdbuf.copyBuffer(_staging_buffer, dst_buffer, 1, &copy);
            });
        }
        _device.get_device().waitIdle();
    }

private:
    Device &_device;
    size_t _staging_buffer_size_mib;
    VmaBuffer _staging_buffer;
};


int main()
{
    auto t0 = std::chrono::high_resolution_clock::now();

    const char *inputfile = "C:/Users/jensw/Downloads/Interior/interior.obj";
    const char *mtl_basedir = "C:/Users/jensw/Downloads/Interior/";
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;
    constexpr bool triangulate = true;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile, mtl_basedir, triangulate);

    if (!warn.empty()) { std::cerr << "Warning: " << warn << "\n"; }
    if (!err.empty()) { std::cerr << "Error: " << warn << "\n"; }
    if (!ret) { return -1; }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Success! This took " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
              << " us\n";


    std::cout << shapes.size() << "\n";
    std::cout << materials.size() << "\n";

    size_t nl = 0;
    size_t np = 0;
    size_t nv = 0;
    size_t onv = 0;
    for (auto &shape : shapes) {
        nl = std::max(shape.lines.num_line_vertices.size(), nl);
        np = std::max(shape.points.indices.size(), np);
        nv = std::max(shape.mesh.num_face_vertices.size(), nv);
        onv += shape.mesh.num_face_vertices.size();
    }

    std::vector<glm::vec3> vertex_positions;
    std::vector<glm::vec3> vertex_normals;
    std::vector<glm::vec2> vertex_texcoords;
    std::vector<uint32_t> vertex_indices;

    // index combination -> unified vertex index
    std::unordered_map<IndexKey, int> vi_mapping;
    for (const auto &shape : shapes) {
        for (const auto &i : shape.mesh.indices) {
            IndexKey key(i.vertex_index, i.normal_index, i.texcoord_index);
            auto iter = vi_mapping.find(key);
            int unified_vi;
            if (iter == vi_mapping.end()) {
                unified_vi = vertex_positions.size();
                vi_mapping[key] = unified_vi;
                vertex_positions.emplace_back(
                    attrib.vertices[i.vertex_index * 3 + 0],
                    attrib.vertices[i.vertex_index * 3 + 1],
                    attrib.vertices[i.vertex_index * 3 + 2]);
                vertex_normals.emplace_back(
                    attrib.normals[i.normal_index * 3 + 0],
                    attrib.normals[i.normal_index * 3 + 1],
                    attrib.normals[i.normal_index * 3 + 2]);
                vertex_texcoords.emplace_back(
                    attrib.texcoords[i.texcoord_index * 2 + 0], attrib.texcoords[i.texcoord_index * 2 + 1]);
            } else {
                unified_vi = iter->second;
            }
            vertex_indices.push_back(unified_vi);
        }
    }

    try {
        //----------------------------------------------------------------------
        // Vulkan instance

        vk::DynamicLoader dl;
        VULKAN_HPP_DEFAULT_DISPATCHER.init(dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));

        std::vector<vk::ExtensionProperties> ext_props = vk::enumerateInstanceExtensionProperties();
        if (true) {
            std::cerr << "Available instance extensions:\n";
            for (const auto &ext_prop : ext_props) { std::cerr << ext_prop.extensionName << "\n"; }
        }

        std::vector<const char *> instance_extensions = {
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME, VK_KHR_SURFACE_EXTENSION_NAME, "VK_KHR_get_surface_capabilities2"};
#if defined(_WIN32)
        instance_extensions.push_back("VK_KHR_win32_surface");
#elif !(defined(__APPLE__) || defined(__MACH__))
        if (std::find_if(
                ext_props.cbegin(),
                ext_props.cend(),
                [](const auto &ep) { return std::string(ep.extensionName) == "VK_KHR_xcb_surface"; })
            != ext_props.cend()) {
            instance_extensions.push_back("VK_KHR_xcb_surface");
        } else {
            instance_extensions.push_back("VK_KHR_xlib_surface");
        }
#else
#    error This platform is not yet supported
#endif

        uint32_t glfw_exts_count = 0;
        const char **glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_exts_count);
        for (uint32_t i = 0; i < glfw_exts_count; ++i) { instance_extensions.push_back(glfw_exts[i]); }

        vk::ApplicationInfo appInfo("Test", 1, "Custom", 1, VK_API_VERSION_1_1);
        std::vector<const char *> layer_names = {
            "VK_LAYER_KHRONOS_validation", "VK_LAYER_LUNARG_monitor",
            //"VK_LAYER_LUNARG_api_dump",
        };
        vk::UniqueInstance instance = vk::createInstanceUnique(vk::InstanceCreateInfo{}
                                                                   .setPApplicationInfo(&appInfo)
                                                                   .setPEnabledExtensionNames(instance_extensions)
                                                                   .setPEnabledLayerNames(layer_names));
        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());

        auto debug_utils_messenger = instance->createDebugUtilsMessengerEXTUnique(
            vk::DebugUtilsMessengerCreateInfoEXT{}
                .setMessageSeverity(
                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
                    | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
                    //| vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
                    | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo)
                .setMessageType(
                    vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
                    | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation)
                .setPfnUserCallback(&debug_utils_callback));

        //----------------------------------------------------------------------
        // Use GLFW3 to create a window and a corresponding Vulkan surface.

        // Init GLFW itself
        ASSUME(glfwInit());
        glfwSetErrorCallback(&glfwErrorCallback);
        ASSUME(glfwVulkanSupported());

        Window window(instance.get(), 1200, 800, "Vulkan Raster");

        /*
        glfwSetKeyCallback(
            window.get_glfw_window(), [](GLFWwindow *window, int key, int scancode, int action, int mods) {
                if (!ImGui::GetIO().WantCaptureKeyboard) {
                    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) { glfwSetWindowShouldClose(window, GLFW_TRUE); }
                }
            });
        glfwSetCursorPosCallback(window.get_glfw_window(), &cursor_position_callback);
        glfwSetMouseButtonCallback(window.get_glfw_window(), &mouse_button_callback);
        glfwSetScrollCallback(window.get_glfw_window(), &scroll_callback);
        */

        //----------------------------------------------------------------------
        // Physical and logical device.

        // Choose first physical device
        std::vector<vk::PhysicalDevice> physical_devices = instance->enumeratePhysicalDevices();
        ASSUME(!physical_devices.empty());

        vk::PhysicalDevice phys_device = physical_devices.front();

        vk::PhysicalDeviceProperties phys_props = phys_device.getProperties();

        std::vector<const char *> device_extensions = {
            "VK_KHR_swapchain", "VK_KHR_get_memory_requirements2", "VK_KHR_dedicated_allocation"};

        Device device(instance.get(), phys_device, window.get_surface());
        window.make_swap_chain(device, true, false);
        // window.make_swap_chain(device, false, false); // Pretty much immediate present

        //----------------------------------------------------------------------
        // Static resources (not dependent on window size)

        vk::UniquePipelineCache pipeline_cache = device.get().createPipelineCacheUnique(vk::PipelineCacheCreateInfo{});

        vk::UniqueDescriptorPool descriptor_pool = device.get().createDescriptorPoolUnique(
            vk::DescriptorPoolCreateInfo{}
                .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet)
                .setMaxSets(100)
                .setPoolSizes(std::initializer_list<vk::DescriptorPoolSize>{
                    vk::DescriptorPoolSize{}.setType(vk::DescriptorType::eStorageImage).setDescriptorCount(100),
                    vk::DescriptorPoolSize{}.setType(vk::DescriptorType::eUniformBuffer).setDescriptorCount(100)}));

        // vk::UniqueSampler image_sampler = device.get().createSamplerUnique(vk::SamplerCreateInfo{});

        vk::UniqueShaderModule vertex_shader =
            load_shader(device.get(), std::string(build_info::PROJECT_BINARY_DIR) + "/shaders/vs.vert.spirv");
        vk::UniqueShaderModule fragment_shader =
            load_shader(device.get(), std::string(build_info::PROJECT_BINARY_DIR) + "/shaders/fs.frag.spirv");

        //----------------------------------------------------------------------
        // Vertex and index buffers

        VmaBuffer vertex_buffer = std::move(VmaBuffer(
            device.get_vma_allocator(),
            VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            false,
            vk::BufferCreateInfo{}
                .setUsage(vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setSize(
                    vertex_positions.size() * sizeof(glm::vec3) + vertex_normals.size() * sizeof(glm::vec3)
                    + vertex_texcoords.size() * sizeof(glm::vec2))));

        VmaBuffer index_buffer = std::move(VmaBuffer(
            device.get_vma_allocator(),
            VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY,
            false,
            vk::BufferCreateInfo{}
                .setUsage(vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setSize(vertex_indices.size() * sizeof(uint32_t))));

        // Upload
        {
            Staging staging(device, 100);
            staging.copy(vertex_positions.data(), vertex_positions.size() * sizeof(glm::vec3), vertex_buffer.buffer, 0);
            staging.copy(
                vertex_normals.data(),
                vertex_normals.size() * sizeof(glm::vec3),
                vertex_buffer.buffer,
                vertex_positions.size() * sizeof(glm::vec3));
            staging.copy(vertex_indices.data(), vertex_indices.size() * sizeof(uint32_t), index_buffer.buffer, 0);
        }

        //----------------------------------------------------------------------
        // Pipeline

        struct GraphicsPipeline
        {
            vk::UniquePipeline pipeline;
            vk::UniquePipelineLayout layout;
            vk::UniqueRenderPass render_pass;
            vk::UniqueDescriptorSetLayout dsl;

            struct PushConstants
            {
                alignas(16) glm::mat4 mvp = glm::mat4(1);
            } push_constants;

            VmaImage depth_buffer;
            vk::UniqueImageView depth_buffer_view;

        } graphics_pipeline;

        auto update_size_dependent_resource = [&](const vk::Extent2D &extent) {
            static vk::Extent2D last_extent;

            if (last_extent == extent) {
                // Nothing to do here.
                return false;
            }
            last_extent = extent;

            {
                // TODO: better logic to find the best supported depth format (see
                // https://vulkan-tutorial.com/Depth_buffering)
                graphics_pipeline.depth_buffer = VmaImage(
                    device.get_vma_allocator(),
                    VMA_MEMORY_USAGE_GPU_ONLY,
                    vk::ImageCreateInfo{}
                        .setImageType(vk::ImageType::e2D)
                        .setExtent(vk::Extent3D(window.get_extent(), 1))
                        .setMipLevels(1)
                        .setArrayLayers(1)
                        .setFormat(vk::Format::eD24UnormS8Uint)
                        .setTiling(vk::ImageTiling::eOptimal)
                        .setInitialLayout(vk::ImageLayout::eUndefined)
                        .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment)
                        .setSamples(vk::SampleCountFlagBits::e1)
                        .setSharingMode(vk::SharingMode::eExclusive));
                graphics_pipeline.depth_buffer_view = device.get().createImageViewUnique(
                    vk::ImageViewCreateInfo{}
                        .setImage(graphics_pipeline.depth_buffer)
                        .setFormat(vk::Format::eD24UnormS8Uint)
                        .setViewType(vk::ImageViewType::e2D)
                        .setSubresourceRange(vk::ImageSubresourceRange{}
                                                 .setAspectMask(vk::ImageAspectFlagBits::eDepth)
                                                 .setBaseMipLevel(0)
                                                 .setLayerCount(1)
                                                 .setLevelCount(1)
                                                 .setBaseArrayLayer(0)));

                std::array<vk::AttachmentDescription, 2> render_pass_attachments = {
                    vk::AttachmentDescription{}
                        .setFormat(window.get_swap_chain().get_format())
                        .setSamples(vk::SampleCountFlagBits::e1)
                        .setLoadOp(vk::AttachmentLoadOp::eClear)
                        .setStoreOp(vk::AttachmentStoreOp::eStore)
                        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                        .setInitialLayout(vk::ImageLayout::eUndefined)
                        //.setInitialLayout(vk::ImageLayout::eColorAttachmentOptimal)  // TODO
                        .setFinalLayout(vk::ImageLayout::ePresentSrcKHR),
                    vk::AttachmentDescription{}
                        .setFormat(vk::Format::eD24UnormS8Uint) // TODO: richer image class to query this automatically
                        .setSamples(vk::SampleCountFlagBits::e1)
                        .setLoadOp(vk::AttachmentLoadOp::eClear)
                        .setStoreOp(vk::AttachmentStoreOp::eDontCare)
                        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                        .setInitialLayout(vk::ImageLayout::eUndefined)
                        .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)};

                std::array<vk::AttachmentReference, 1> color_ar = {
                    vk::AttachmentReference{}.setAttachment(0).setLayout(vk::ImageLayout::eColorAttachmentOptimal)};

                auto depth_ar = vk::AttachmentReference{}.setAttachment(1).setLayout(
                    vk::ImageLayout::eDepthStencilAttachmentOptimal);

                std::array<vk::SubpassDescription, 1> subpass_desc = {
                    vk::SubpassDescription{}
                        .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                        .setColorAttachments(color_ar)
                        .setPDepthStencilAttachment(&depth_ar)};

                // Barrier around depth buffer usage -- otherwise multiple frames in flight
                // could access it at the same time. See https://stackoverflow.com/a/62398311

                std::array<vk::SubpassDependency, 2> subpass_deps = {
                    vk::SubpassDependency{}
                        .setSrcSubpass(VK_SUBPASS_EXTERNAL)
                        .setSrcStageMask(
                            vk::PipelineStageFlagBits::eEarlyFragmentTests
                            | vk::PipelineStageFlagBits::eLateFragmentTests)
                        .setSrcAccessMask(vk::AccessFlagBits::eDepthStencilAttachmentWrite)
                        .setDstSubpass(0)
                        .setDstStageMask(
                            vk::PipelineStageFlagBits::eEarlyFragmentTests
                            | vk::PipelineStageFlagBits::eLateFragmentTests)
                        .setDstAccessMask(
                            vk::AccessFlagBits::eDepthStencilAttachmentRead
                            | vk::AccessFlagBits::eDepthStencilAttachmentWrite),
                    vk::SubpassDependency{}
                        .setSrcSubpass(0)
                        .setSrcStageMask(vk::PipelineStageFlagBits::eAllGraphics)
                        .setSrcAccessMask(
                            vk::AccessFlagBits::eColorAttachmentWrite
                            | vk::AccessFlagBits::eDepthStencilAttachmentWrite)
                        .setDstSubpass(VK_SUBPASS_EXTERNAL)
                        .setDstStageMask(vk::PipelineStageFlagBits::eBottomOfPipe)};

                graphics_pipeline.render_pass =
                    device.get().createRenderPassUnique(vk::RenderPassCreateInfo{}
                                                            .setAttachments(render_pass_attachments)
                                                            .setSubpasses(subpass_desc)
                                                            .setDependencies(subpass_deps));

                std::vector<vk::PipelineShaderStageCreateInfo> shader_stages = {
                    vk::PipelineShaderStageCreateInfo{}
                        .setStage(vk::ShaderStageFlagBits::eVertex)
                        .setModule(vertex_shader.get())
                        .setPName("main"),
                    vk::PipelineShaderStageCreateInfo{}
                        .setStage(vk::ShaderStageFlagBits::eFragment)
                        .setModule(fragment_shader.get())
                        .setPName("main")};


                std::array<vk::VertexInputBindingDescription, 2> vibds = {
                    vk::VertexInputBindingDescription{}
                        .setBinding(0)
                        .setStride(sizeof(glm::vec3))
                        .setInputRate(vk::VertexInputRate::eVertex),
                    vk::VertexInputBindingDescription{}
                        .setBinding(1)
                        .setStride(sizeof(glm::vec3))
                        .setInputRate(vk::VertexInputRate::eVertex)};

                std::array<vk::VertexInputAttributeDescription, 2> viads = {
                    vk::VertexInputAttributeDescription{}
                        .setBinding(0)
                        .setLocation(0)
                        .setFormat(vk::Format::eR32G32B32Sfloat)
                        .setOffset(0),
                    vk::VertexInputAttributeDescription{}
                        .setBinding(1)
                        .setLocation(1)
                        .setFormat(vk::Format::eR32G32B32Sfloat)
                        .setOffset(0)};

                auto vertex_input_state = vk::PipelineVertexInputStateCreateInfo{}
                                              .setVertexBindingDescriptions(vibds)
                                              .setVertexAttributeDescriptions(viads);

                auto input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo{}
                                                .setTopology(vk::PrimitiveTopology::eTriangleList)
                                                .setPrimitiveRestartEnable(false);

                std::array<vk::DynamicState, 2> dyn_state_flags = {
                    vk::DynamicState::eViewport, vk::DynamicState::eScissor};
                auto dyn_state = vk::PipelineDynamicStateCreateInfo{}.setDynamicStates(dyn_state_flags);

                auto vp = vk::Viewport(0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1);
                auto scissor_rect = vk::Rect2D({0, 0}, extent);
                auto viewport_state = vk::PipelineViewportStateCreateInfo{}.setViewports(vp).setScissors(scissor_rect);

                auto rasterization_state = vk::PipelineRasterizationStateCreateInfo{}
                                               .setPolygonMode(vk::PolygonMode::eFill)
                                               .setCullMode(vk::CullModeFlagBits::eNone)
                                               .setFrontFace(vk::FrontFace::eClockwise)
                                               .setDepthClampEnable(false)
                                               .setRasterizerDiscardEnable(false)
                                               .setDepthBiasEnable(false)
                                               .setLineWidth(1.f);
                auto multisample_state = vk::PipelineMultisampleStateCreateInfo{}
                                             .setRasterizationSamples(vk::SampleCountFlagBits::e1)
                                             .setSampleShadingEnable(false);
                auto depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo{}
                                               .setDepthTestEnable(true)
                                               .setDepthWriteEnable(true)
                                               //.setDepthTestEnable(false) // HACK
                                               //.setDepthWriteEnable(false) // HACK
                                               .setDepthCompareOp(vk::CompareOp::eLessOrEqual)
                                               .setBack(vk::StencilOpState{}
                                                            .setFailOp(vk::StencilOp::eKeep)
                                                            .setPassOp(vk::StencilOp::eKeep)
                                                            .setCompareOp(vk::CompareOp::eAlways))
                                               .setFront(vk::StencilOpState{}
                                                             .setFailOp(vk::StencilOp::eKeep)
                                                             .setPassOp(vk::StencilOp::eKeep)
                                                             .setCompareOp(vk::CompareOp::eAlways));

                auto cbas = vk::PipelineColorBlendAttachmentState{}.setColorWriteMask(
                    vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB
                    | vk::ColorComponentFlagBits::eA);
                auto color_blend_state = vk::PipelineColorBlendStateCreateInfo{}.setAttachments(cbas);

                graphics_pipeline.dsl =
                    device.get().createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{});

                graphics_pipeline.layout = device.get().createPipelineLayoutUnique(
                    vk::PipelineLayoutCreateInfo{}
                        .setPushConstantRanges(vk::PushConstantRange{}
                                                   .setOffset(0)
                                                   .setSize(sizeof(GraphicsPipeline::PushConstants))
                                                   .setStageFlags(vk::ShaderStageFlagBits::eVertex))
                        .setSetLayouts(graphics_pipeline.dsl.get()));

                graphics_pipeline.pipeline = device.get().createGraphicsPipelineUnique(
                    pipeline_cache.get(),
                    vk::GraphicsPipelineCreateInfo{}
                        .setLayout(graphics_pipeline.layout.get())
                        .setRenderPass(graphics_pipeline.render_pass.get())
                        .setStages(shader_stages)
                        .setPVertexInputState(&vertex_input_state)
                        .setPInputAssemblyState(&input_assembly_state)
                        .setPDynamicState(&dyn_state)
                        .setPViewportState(&viewport_state)
                        .setPRasterizationState(&rasterization_state)
                        .setPMultisampleState(&multisample_state)
                        .setPDepthStencilState(&depth_stencil_state)
                        .setPColorBlendState(&color_blend_state));

                return true;
            }
        };

        //----------------------------------------------------------------------


        //----------------------------------------------------------------------
        // Render loop

        while (!window.should_close()) {
            auto this_time = std::chrono::high_resolution_clock::now();
            glfwPollEvents();

            SwapChain::FrameImage frame = window.get_swap_chain().begin_next_frame();
            if (frame.is_valid()) {
                update_size_dependent_resource(window.get_extent());

                FramebufferKey fb_key;
                fb_key.render_pass = graphics_pipeline.render_pass.get();
                fb_key.extent = window.get_extent();
                fb_key.attachments.push_back(frame.get_image_view());
                fb_key.attachments.push_back(graphics_pipeline.depth_buffer_view.get());

                std::array<vk::ClearValue, 2> clear_values = {
                    vk::ClearValue{}.setColor(vk::ClearColorValue{}.setFloat32({0.5, 1, 0, 1})),
                    vk::ClearValue{}.setDepthStencil(vk::ClearDepthStencilValue{}.setDepth((1.0)))};

                vk::CommandBuffer cmd_buffer = frame.get_cmd_buffer(Device::Queue::Graphics);
                cmd_buffer.beginRenderPass(
                    vk::RenderPassBeginInfo{}
                        .setRenderPass(graphics_pipeline.render_pass.get())
                        .setRenderArea(vk::Rect2D({0, 0}, window.get_extent()))
                        .setClearValues(clear_values)
                        .setFramebuffer(device.get_framebuffer(fb_key)),
                    vk::SubpassContents::eInline);

#if 0
            // We do this as part of the render pass begin, but here is manual clearing for completeness.
                auto ca = vk::ClearAttachment{}
                              .setClearValue(vk::ClearValue{}.setColor(vk::ClearColorValue{}.setFloat32({0, 0, 1, 1})))
                              .setColorAttachment(0)
                              .setAspectMask(vk::ImageAspectFlagBits::eColor);
                auto cr = vk::ClearRect{}.setBaseArrayLayer(0).setLayerCount(1).setRect(
                    vk::Rect2D({0, 0}, window.get_extent()));
                cmd_buffer.clearAttachments(1, &ca, 1, &cr);
#endif

                cmd_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline.pipeline.get());

                graphics_pipeline.push_constants.mvp =
                    glm::perspective(80.0f, window.get_width() / float(window.get_height()), 0.1f, 1000.0f)
                    * glm::lookAt(glm::vec3(880, 100, -100), glm::vec3(880 - 10, 100, -100), glm::vec3(0, 1, 0));

                cmd_buffer.pushConstants<GraphicsPipeline::PushConstants>(
                    graphics_pipeline.layout.get(),
                    vk::ShaderStageFlagBits::eVertex,
                    0,
                    graphics_pipeline.push_constants);

                // TODO: not actually dynamic state -- we recreate this pipeline when extent changes anyway.
                cmd_buffer.setViewport(
                    0,
                    vk::Viewport(
                        0, 0, static_cast<float>(window.get_width()), static_cast<float>(window.get_height()), 0, 1));
                cmd_buffer.setScissor(0, vk::Rect2D({0, 0}, window.get_extent()));

                cmd_buffer.bindIndexBuffer(index_buffer, 0, vk::IndexType::eUint32);

                {
                    std::array<vk::Buffer, 2> vbs = {vertex_buffer, vertex_buffer};
                    std::array<vk::DeviceSize, 2> offsets = {0, vertex_positions.size() * sizeof(glm::vec3)};
                    cmd_buffer.bindVertexBuffers(0, vbs, offsets);
                }

                cmd_buffer.drawIndexed(to_uint32(vertex_indices.size()), 1, 0, 0, 0);

                cmd_buffer.endRenderPass();
            }
        }

        device.get().waitIdle();

        // We must do this before the Device is destroyed.
        window.destroy_swap_chain();
    } catch (vk::SystemError &err) {
        std::cerr << "vk::SystemError: " << err.what() << "\n";
        exit(-1);
    } catch (std::exception &err) {
        std::cerr << "std::exception: " << err.what() << "\n";
        exit(-1);
    } catch (...) {
        std::cerr << "Unknown exception!\n";
        exit(-1);
    }

    return 0;
}
