# shader.py

from pyglet.graphics.shader import Shader, ShaderProgram

VERTEX_SHADER_SRC = """
#version 150 core

in vec2 position;
in vec3 color;

out vec3 frag_color;

uniform mat4 projection;

void main()
{
    frag_color = color;
    gl_Position = projection * vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER_SRC = """
#version 150 core

in vec3 frag_color;
out vec4 final_color;

void main()
{
    final_color = vec4(frag_color, 1.0);
}
"""

def get_default_shader():
    """
    Compile and return the default shader program used for rendering colored 2D geometry.
    
    Returns:
        ShaderProgram: A compiled and linked shader program.
    """
    vertex_shader = Shader(VERTEX_SHADER_SRC, 'vertex')
    fragment_shader = Shader(FRAGMENT_SHADER_SRC, 'fragment')
    return ShaderProgram(vertex_shader, fragment_shader)
