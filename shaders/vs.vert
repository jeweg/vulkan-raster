#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 in_pos;

layout(location = 0) out vec3 out_color;

layout(push_constant) uniform PushConstantBlock
{
    mat4 mvp;
} PushConstants;

void main()
{
    gl_Position = PushConstants.mvp * vec4(in_pos, 1.0);
    out_color = vec3(sin(gl_VertexIndex), sin(gl_VertexIndex * 7.31 + 2.7), sin(gl_VertexIndex * 19.3 + 4)) * 0.4 + 0.4;
}

