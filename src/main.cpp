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
    if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) { debug_break(); }
    return VK_TRUE;
}


void glfwErrorCallback(int error, const char *description)
{
    std::cerr << "GLFW error " << std::to_string(error) << ": " << description << "\n";
}


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

    struct Vertex
    {
        glm::vec3 pos;
    };

    std::vector<Vertex> vertices;
    vertices.resize(10000);
    uint32_t max_vi = 0;
    std::vector<uint32_t> indices;
    glm::vec3 vertices_bbox_min(std::numeric_limits<float>::max());
    glm::vec3 vertices_bbox_max(std::numeric_limits<float>::lowest());
    indices.reserve(vertices.size() * 3);
    for (const auto &shape : shapes) {
        for (const auto &i : shape.mesh.indices) {
            uint32_t vi = i.vertex_index;
            indices.push_back(vi);
            Vertex vertex;
            vertex.pos =
                glm::vec3(attrib.vertices[vi * 3 + 0], attrib.vertices[vi * 3 + 1], attrib.vertices[vi * 3 + 2]);

            vertices_bbox_min = glm::min(vertices_bbox_min, vertex.pos);
            vertices_bbox_max = glm::max(vertices_bbox_min, vertex.pos);

            if (vi >= vertices.size()) { vertices.resize(vertices.size() * 3 / 2); }
            max_vi = std::max(max_vi, vi);
            vertices[vi] = std::move(vertex);
        }
    }
    vertices.resize(max_vi + 1);


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
#    error Currently unsupported platform!
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
                    | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
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
        // window.make_swap_chain(device, true, true);
        window.make_swap_chain(device, false, false); // Pretty much immediate present

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
        // Vertex and index buffer

        VmaBuffer vertex_buffer = std::move(VmaBuffer(
            device.get_vma_allocator(),
            VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            true, // Automatically persistently mapped
            vk::BufferCreateInfo{}
                .setUsage(vk::BufferUsageFlagBits::eVertexBuffer)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setSize(vertices.size() * sizeof(Vertex))));
        std::memcpy(vertex_buffer.mapped_data(), vertices.data(), vertices.size() * sizeof(Vertex));

        VmaBuffer index_buffer = std::move(VmaBuffer(
            device.get_vma_allocator(),
            VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU,
            true, // Automatically persistently mapped
            vk::BufferCreateInfo{}
                .setUsage(vk::BufferUsageFlagBits::eIndexBuffer)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setSize(indices.size() * sizeof(uint32_t))));
        std::memcpy(index_buffer.mapped_data(), indices.data(), indices.size() * sizeof(uint32_t));

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

                graphics_pipeline.render_pass = device.get().createRenderPassUnique(
                    vk::RenderPassCreateInfo{}
                        .setAttachments(vk::AttachmentDescription{}
                                            .setFormat(window.get_swap_chain().get_format())
                                            .setSamples(vk::SampleCountFlagBits::e1)
                                            .setLoadOp(vk::AttachmentLoadOp::eClear)
                                            //.setLoadOp(vk::AttachmentLoadOp::eDontCare)
                                            .setStoreOp(vk::AttachmentStoreOp::eStore)
                                            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
                                            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
                                            .setInitialLayout(vk::ImageLayout::eUndefined)
                                            .setFinalLayout(vk::ImageLayout::ePresentSrcKHR))
                        .setSubpasses(vk::SubpassDescription{}
                                          .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
                                          .setColorAttachments(vk::AttachmentReference{}.setAttachment(0).setLayout(
                                              vk::ImageLayout::eColorAttachmentOptimal))));

                std::vector<vk::PipelineShaderStageCreateInfo> shader_stages = {
                    vk::PipelineShaderStageCreateInfo{}
                        .setStage(vk::ShaderStageFlagBits::eVertex)
                        .setModule(vertex_shader.get())
                        .setPName("main"),
                    vk::PipelineShaderStageCreateInfo{}
                        .setStage(vk::ShaderStageFlagBits::eFragment)
                        .setModule(fragment_shader.get())
                        .setPName("main")};

                auto vibd = vk::VertexInputBindingDescription{}
                                .setBinding(0)
                                .setStride(sizeof(Vertex))
                                .setInputRate(vk::VertexInputRate::eVertex);

                auto viad = vk::VertexInputAttributeDescription{}
                                .setBinding(0)
                                .setLocation(0)
                                .setFormat(vk::Format::eR32G32B32Sfloat)
                                .setOffset(offsetof(Vertex, pos));

                auto vertex_input_state = vk::PipelineVertexInputStateCreateInfo{}
                                              .setVertexBindingDescriptions({1, &vibd})
                                              .setVertexAttributeDescriptions({1, &viad});

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
                auto depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo{} //.setDepthTestEnable(true)
                                                                                     //.setDepthWriteEnable(true)
                                               .setDepthTestEnable(false) // HACK
                                               .setDepthWriteEnable(false) // HACK
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

                vk::CommandBuffer cmd_buffer = frame.get_cmd_buffer(Device::Queue::Graphics);
                cmd_buffer.beginRenderPass(
                    vk::RenderPassBeginInfo{}
                        .setRenderPass(graphics_pipeline.render_pass.get())
                        .setRenderArea(vk::Rect2D({0, 0}, window.get_extent()))
                        .setClearValues(vk::ClearValue{}
                                            .setColor(vk::ClearColorValue{}.setFloat32({0.5, 1, 0, 1}))
                                            .setDepthStencil(vk::ClearDepthStencilValue{}.setDepth((1.0))))
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
                    * glm::lookAt(glm::vec3(880, 100, -800), glm::vec3(880 - 10, 100, -800), glm::vec3(0, 1, 0));

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
                std::array<vk::Buffer, 1> vbs = {vertex_buffer};
                std::array<vk::DeviceSize, 1> offsets = {0};
                cmd_buffer.bindVertexBuffers(0, 1, vbs.data(), offsets.data());

                // cmd_buffer.draw(3, 1, 0, 0);
                cmd_buffer.drawIndexed(indices.size(), 1, 0, 0, 0);

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


#if 0

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
        std::cout << "Success! That took " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
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

        auto &shape = shapes[0];
#endif
