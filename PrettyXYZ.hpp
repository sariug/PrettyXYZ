#pragma once
#ifndef PRETTY_NO_GLEW
#include <GL/glew.h>
#endif
#include <GL/gl.h>

#ifdef PRETTY_TEXT_RENDER
#include <ft2build.h>
#include FT_FREETYPE_H
#include <map>
#endif

#include <algorithm>
#include <cmath>    // trigonometry
#include <iostream>
#include <vector>

namespace PrettyXYZ {
namespace ShaderProgramCheck {
void checkShader(GLuint);
void checkShader(GLuint);
void checkShader(GLuint);
void checkProgram(GLuint);
} // namespace ShaderProgramCheck
enum class STYLE { LINE, LINE_WITH_HEAD, CYLINDER, CYLINDER_WITH_HEAD, CONE };
namespace MathUtils {
constexpr float pi = 3.14159265;
// Declerations
// Base
class Vector3;
class Vector4;
class Matrix4;
//Arithmatics
inline float dot(const Vector4 &lhs, const Vector4 &rhs);
inline Vector3 operator+(const Vector3 &lhs, const Vector3 &rhs);
inline void operator+=(Vector3 &lhs, const Vector3 &rhs);
inline Vector3 operator*(const Vector3 &v, float t);
inline Matrix4 operator+(const Matrix4 &lhs, const Matrix4 &rhs);
inline Matrix4 operator*(const Matrix4 &m, const float &f);
inline Vector4 operator*(const Matrix4 &m, const Vector4 &v);
inline Matrix4 operator*(const Matrix4 &m1, const Matrix4 &m2);
inline Matrix4 rotate(const float degrees, const Vector3 axis);

// Defitinitions
class Vector3 {
public:
  Vector3() = default;
  Vector3(float _s) : x(_s), y(_s), z(_s) {}
  Vector3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
  float length() { return sqrtf(x * x + y * y + z * z); }
  void normalized() {
    auto l = this->length();
    assert(l!=0);
    x /= l;
    y /= l;
    z /= l;
  }
  float x{0};
  float y{0};
  float z{0};
};

class Vector4 {
public:
  Vector4() = default;
  Vector4(float _s) : x(_s), y(_s), z(_s), w(_s) {}
  Vector4(float _x, float _y, float _z, float _w)
      : x(_x), y(_y), z(_z), w(_w) {}
  Vector4(Vector3 _o, float _w) : x(_o.x), y(_o.y), z(_o.z), w(_w) {}
  float length() { return sqrtf(x * x + y * y + z * z + w * w); }
  void normalized() {
    auto l = this->length();
    x /= l;
    y /= l;
    z /= l;
    w /= l;
  }
  Vector3 xyz() { return Vector3(x, y, z); }
  float x{0};
  float y{0};
  float z{0};
  float w{0};
};

class Matrix4 {
public:
  Matrix4() {
    // Create Unit matrix
    for (int i = 0; i < 16; i++)
      data[i] = i % 5 == 0 ? 1.0f : 0.0f;
  };
  Matrix4(float n) {
    for (int i = 0; i < 16; i++)
      data[i] = n;
  }
  Matrix4(const Matrix4 &other) {
    std::copy(other.data, other.data+16, data);
  }
  Matrix4(const float *raw) {
    for (int i = 0; i < 16; i++)
      data[i] = raw[i];
  }
  Matrix4(std::initializer_list<float> il) {
    std::copy(il.begin(), il.end(), data);
    // std::iota(indices.begin(), indices.end(), 0);

    // v.insert(v.end(), l.begin(), l.end());
  };
  const float *begin() { return &(data[0]); }
  Vector4 getCol(int i) const {
    if (i < 0 || i > 3)
      throw "Unexpected indice for matrix column";
    return Vector4(data[4 * i], data[4 * i + 1], data[4 * i + 2],
                   data[4 * i + 3]);
  }
  Vector4 getRow(int i) const {
    if (i < 0 || i > 3)
      throw "Unexpected indice for matrix row";
    return Vector4(data[i], data[i + 4], data[i + 8], data[i + 12]);
  }

  void resetTranslation() {
    for (int i = 0; i < 3; i++)
      data[12 + i] = 0;
  }
  void inverted() {
    float inv[16], det;
    int i;

    inv[0] = data[5] * data[10] * data[15] - data[5] * data[11] * data[14] -
             data[9] * data[6] * data[15] + data[9] * data[7] * data[14] +
             data[13] * data[6] * data[11] - data[13] * data[7] * data[10];

    inv[4] = -data[4] * data[10] * data[15] + data[4] * data[11] * data[14] +
             data[8] * data[6] * data[15] - data[8] * data[7] * data[14] -
             data[12] * data[6] * data[11] + data[12] * data[7] * data[10];

    inv[8] = data[4] * data[9] * data[15] - data[4] * data[11] * data[13] -
             data[8] * data[5] * data[15] + data[8] * data[7] * data[13] +
             data[12] * data[5] * data[11] - data[12] * data[7] * data[9];

    inv[12] = -data[4] * data[9] * data[14] + data[4] * data[10] * data[13] +
              data[8] * data[5] * data[14] - data[8] * data[6] * data[13] -
              data[12] * data[5] * data[10] + data[12] * data[6] * data[9];

    inv[1] = -data[1] * data[10] * data[15] + data[1] * data[11] * data[14] +
             data[9] * data[2] * data[15] - data[9] * data[3] * data[14] -
             data[13] * data[2] * data[11] + data[13] * data[3] * data[10];

    inv[5] = data[0] * data[10] * data[15] - data[0] * data[11] * data[14] -
             data[8] * data[2] * data[15] + data[8] * data[3] * data[14] +
             data[12] * data[2] * data[11] - data[12] * data[3] * data[10];

    inv[9] = -data[0] * data[9] * data[15] + data[0] * data[11] * data[13] +
             data[8] * data[1] * data[15] - data[8] * data[3] * data[13] -
             data[12] * data[1] * data[11] + data[12] * data[3] * data[9];

    inv[13] = data[0] * data[9] * data[14] - data[0] * data[10] * data[13] -
              data[8] * data[1] * data[14] + data[8] * data[2] * data[13] +
              data[12] * data[1] * data[10] - data[12] * data[2] * data[9];

    inv[2] = data[1] * data[6] * data[15] - data[1] * data[7] * data[14] -
             data[5] * data[2] * data[15] + data[5] * data[3] * data[14] +
             data[13] * data[2] * data[7] - data[13] * data[3] * data[6];

    inv[6] = -data[0] * data[6] * data[15] + data[0] * data[7] * data[14] +
             data[4] * data[2] * data[15] - data[4] * data[3] * data[14] -
             data[12] * data[2] * data[7] + data[12] * data[3] * data[6];

    inv[10] = data[0] * data[5] * data[15] - data[0] * data[7] * data[13] -
              data[4] * data[1] * data[15] + data[4] * data[3] * data[13] +
              data[12] * data[1] * data[7] - data[12] * data[3] * data[5];

    inv[14] = -data[0] * data[5] * data[14] + data[0] * data[6] * data[13] +
              data[4] * data[1] * data[14] - data[4] * data[2] * data[13] -
              data[12] * data[1] * data[6] + data[12] * data[2] * data[5];

    inv[3] = -data[1] * data[6] * data[11] + data[1] * data[7] * data[10] +
             data[5] * data[2] * data[11] - data[5] * data[3] * data[10] -
             data[9] * data[2] * data[7] + data[9] * data[3] * data[6];

    inv[7] = data[0] * data[6] * data[11] - data[0] * data[7] * data[10] -
             data[4] * data[2] * data[11] + data[4] * data[3] * data[10] +
             data[8] * data[2] * data[7] - data[8] * data[3] * data[6];

    inv[11] = -data[0] * data[5] * data[11] + data[0] * data[7] * data[9] +
              data[4] * data[1] * data[11] - data[4] * data[3] * data[9] -
              data[8] * data[1] * data[7] + data[8] * data[3] * data[5];

    inv[15] = data[0] * data[5] * data[10] - data[0] * data[6] * data[9] -
              data[4] * data[1] * data[10] + data[4] * data[2] * data[9] +
              data[8] * data[1] * data[6] - data[8] * data[2] * data[5];

    det = data[0] * inv[0] + data[1] * inv[4] + data[2] * inv[8] +
          data[3] * inv[12];

    if (det == 0)
      return;

    det = 1.0f / det;

    for (i = 0; i < 16; i++)
      data[i] = inv[i] * det;
  }
  float &operator[](int i) {
    if (i >= 16) {
      std::cout << "Array index out of bound, exiting";
      throw;
    }
    return data[i];
  }
  const float operator[](int i) const {
    if (i >= 16) {
      std::cout << "Array index out of bound, exiting";
      throw;
    }
    return data[i];
  }

private:
  // store columnwise !
  /*
  [0,0][1,0][2,0][3,0]
  [0,1][1,1][2,1][3,1]
  [0,2][1,2][2,2][3,2]
  [0,3][1,3][2,3][3,3]
  */
  float data[16];
};

// Arithmatics
inline float dot(const Vector4 &lhs, const Vector4 &rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w;
}
inline Vector3 operator+(const Vector3 &lhs, const Vector3 &rhs) {
  return Vector3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
};
inline void operator+=(Vector3 &lhs, const Vector3 &rhs) {
  lhs.x += rhs.x;
  lhs.y += rhs.y;
  lhs.z += rhs.z;
};
inline Vector3 operator*(const Vector3 &v, float t) {
  return Vector3(v.x * t, v.y * t, v.z * t);
};
inline Matrix4 operator+(const Matrix4 &lhs, const Matrix4 &rhs) {
  Matrix4 result;
  for (int i = 0; i < 16; i++)
    result[i] = lhs[i] + rhs[i];
  return result;
}
inline Matrix4 operator*(const Matrix4 &m, const float &f) {
  Matrix4 result;
  for (int i = 0; i < 16; i++)
    result[i] = m[i] * f;
  return result;
}
inline Vector4 operator*(const Matrix4 &m, const Vector4 &v) {
  return Vector4(m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12] * v.w,
                 m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13] * v.w,
                 m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w,
                 m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w);
}
inline Matrix4 operator*(const Matrix4 &m1, const Matrix4 &m2) {
  Matrix4 result(0.0);
  result[0] = dot(m1.getRow(0), m2.getCol(0));
  result[4] = dot(m1.getRow(0), m2.getCol(1));
  result[8] = dot(m1.getRow(0), m2.getCol(2));
  result[12] = dot(m1.getRow(0), m2.getCol(3));
  result[1] = dot(m1.getRow(1), m2.getCol(0));
  result[5] = dot(m1.getRow(1), m2.getCol(1));
  result[9] = dot(m1.getRow(1), m2.getCol(2));
  result[13] = dot(m1.getRow(1), m2.getCol(3));
  result[2] = dot(m1.getRow(2), m2.getCol(0));
  result[6] = dot(m1.getRow(2), m2.getCol(1));
  result[10] = dot(m1.getRow(2), m2.getCol(2));
  result[14] = dot(m1.getRow(2), m2.getCol(3));
  result[3] = dot(m1.getRow(3), m2.getCol(0));
  result[7] = dot(m1.getRow(3), m2.getCol(1));
  result[11] = dot(m1.getRow(3), m2.getCol(2));
  result[15] = dot(m1.getRow(3), m2.getCol(3));
  return result;
};

inline Matrix4 rotate(const float degrees, const Vector3 axis) {
  float theta = degrees * pi / 180;

  float x = axis.x;
  float y = axis.y;
  float z = axis.z;

  float c = cos(theta);
  float s = sin(theta);

  Matrix4 rotationMat({x * x * (1.0f - c) + c, y * x * (1.0f - c) + z * s,
                       z * x * (1.0f - c) - y * s, 0.0f,
                       x * y * (1.0f - c) - z * s, y * y * (1.0f - c) + c,
                       z * y * (1.0f - c) + x * s, 0.0f,
                       x * z * (1.0f - c) + y * s, y * z * (1.0f - c) - x * s,
                       z * z * (1.0f - c) + c, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f});

  return rotationMat;
}
} // namespace MathUtils
using Vector3 = MathUtils::Vector3;
using Vector4 = MathUtils::Vector4;
using Matrix4 = MathUtils::Matrix4;

struct PrettyVrtx;
struct Character;

/**
 * Creates the coordinate axes according to defined parameters.
 *
 *
 * @param _camera     : 16 float camera(view) matrix. Expected to be columnwise as following:  
 *                
 *                      [0,0][1,0][2,0][3,0]
 *                      [0,1][1,1][2,1][3,1]
 *                      [0,2][1,2][2,2][3,2]
 *                      [0,3][1,3][2,3][3,3] <- Translation vector.
 *
 * @param position    : Position of the origin of the axis on screen space. The third parameter
 *                      shall be 0.
 * @param arrow_size  : The size of the arrow
 * @param render_style: The style of the axis: See STYLE Enums.
 * @param render_text : Bool to render text or not. 
 * @param color_axis_x: Color of the x axis. Default red. 
 * @param color_axis_y: Color of the y axis. Default green. 
 * @param color_axis_z: Color of the z axis. Default blue.
 * @param axis_x      : Text written for axis X. Default "X"
 * @param axis_y      : Text written for axis Y. Default "Y"
 * @param axis_z      : Text written for axis Z. Default "Z"
 * @param color_text_x: Color of the text of x axis. Default red. 
 * @param color_text_y: Color of the text of y axis. Default green. 
 * @param color_text_z: Color of the text of z axis. Default blue. 
 */
void prettyCoordinateAxes(const float *_camera, Vector3 position,
                          float arrow_size, STYLE render_style, bool render_text,
                          Vector3 color_axis_x, Vector3 color_axis_y,
                          Vector3 color_axis_z, const char *axis_x,
                          const char *axis_y, const char *axis_z,
                          Vector3 color_text_x, Vector3 color_text_y,
                          Vector3 color_text_z);
void initGeometryShaderProgram();
void initTextShaderProgram();
void setOrthographicProjectionMatrix();
void updateIndicesForCylinder(std::vector<unsigned int> &indices,
                              int sectorCount);
void renderGeometry(const std::vector<PrettyVrtx> &vertices,
                    std::vector<unsigned int> &indices);
void renderText(std::string text, float scale, Vector3 location, Vector3 color);

struct PrettyVrtx {
  PrettyVrtx() {}
  PrettyVrtx(Vector3 p, Vector3 c) : pos(p), col(c) {}
  PrettyVrtx(Vector3 p, Vector3 n, Vector3 c) : pos(p), normal(n), col(c) {}
  Vector3 pos;
  Vector3 normal = Vector3(0.0f, 1.0f, 0.0f);
  Vector3 col; // Alpha is not supported.
};
struct Character {
  unsigned int TextureID; // ID handle of the glyph texture
  // Todo is this a bug ?
  unsigned int Size[2]; // Size of glyph
  unsigned int Advance; // Horizontal offset to advance to next glyph
};

static Matrix4 prettyOrthoProjection;

// Geometry Shader handle
static GLuint prettyHandleGeoProgram = 0;
static GLuint prettyGeoVao, prettyGeoVbo, prettyGeoIbo;

// Glyph shader handle
static GLuint prettyHandleTextProgram;
static GLuint prettyTextVao, prettyTextVbo;
#ifdef PRETTY_TEXT_RENDER
static std::map<GLchar, Character> Characters;
#endif
const GLchar *text_vs = "#version 410\n"
                        "\n"
                        "layout (location = 0) in vec3 VertexPosition;\n"
                        "\n"
                        "\n"
                        "uniform mat4 projection;\n"
                        "\n"
                        "void main()\n"
                        "{\n"
                        "    gl_Position = vec4(VertexPosition,1.0);\n"
                        "}";

const GLchar *text_gs =
    "#version 410\n"
    "\n"
    "layout( points ) in;\n"
    "layout( triangle_strip, max_vertices = 4 ) out;\n"
    "\n"
    "uniform float width;"
    "uniform float height;"
    "uniform mat4 projection;\n"
    "\n"
    "out vec2 TexCoord;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    mat4 m = projection;\n"
    "\n"
    "    gl_Position = m * (vec4(-width/2,height,0.0,0.0) + "
    "gl_in[0].gl_Position);\n"
    "    TexCoord = vec2(0.0,0.0);\n"
    "    EmitVertex();\n"
    "\n"
    "    gl_Position = m * (vec4(width/2,height,0.0,0.0) + "
    "gl_in[0].gl_Position);\n"
    "    TexCoord = vec2(1.0,0.0);\n"
    "    EmitVertex();\n"
    "\n"
    "    gl_Position = m * (vec4(-width/2,0,0.0,0.0) + gl_in[0].gl_Position);\n"
    "    TexCoord = vec2(0.0,1.0);\n"
    "    EmitVertex();\n"
    "\n"
    "    gl_Position = m * (vec4(width/2,0,0.0,0.0) + gl_in[0].gl_Position);\n"
    "    TexCoord = vec2(1.0,1.0);\n"
    "    EmitVertex();\n"
    "\n"
    "    EndPrimitive();\n"
    "}";

const GLchar *text_fs =

    "#version 410\n"
    "in vec2 TexCoord;\n"
    "\n"
    "uniform sampler2D SpriteTex;\n"
    "uniform vec3 textColor;\n"
    "\n"
    "layout( location = 0 ) out vec4 FragColor;\n"
    "\n"
    "void main()\n"
    "{\n"
    "FragColor = vec4(textColor,texture(SpriteTex, TexCoord).r);\n"
    "}";

const GLchar *shaded_vs = "#version 410\n"
                          "layout (location = 0) in vec3 pos;\n"
                          "layout (location = 1) in vec3 normal;\n"
                          "layout (location = 2) in vec3 col;\n"
                          "uniform mat4 projection;\n"
                          "out vec3 color;\n"
                          "out vec3 norm;\n"
                          "void main()\n"
                          "{\n"
                          "    color = col;\n"
                          "    norm = normal;\n"
                          "    gl_Position =projection*vec4(pos.xyz,1);\n"
                          "}\n";
const GLchar *shaded_fs =
    "#version 410\n"
    "in vec3 color;\n"
    "in vec3 norm;\n"
    "layout (location = 0) out vec4 outColor;\n"
    "void main()\n"
    "{\n"
    "    outColor = vec4(color*max(dot(vec3(1, 1,1),norm)*0.8,0.3),1.0f);\n"
    "}\n";
const GLchar *unrealistic_vs = "#version 410\n"
                               "layout (location = 0) in vec3 pos;\n"
                               "layout (location = 1) in vec3 normal;\n"
                               "layout (location = 2) in vec3 col;\n"
                               "uniform mat4 projection;\n"
                               "out vec3 color;\n"
                               "out float norm;\n"
                               "void main()\n"
                               "{\n"
                               "    color = col;\n"
                               "    float norm = dot(-normalize(pos),normal);\n"
                               "    gl_Position =projection*vec4(pos.xyz,1);\n"
                               "}\n";
const GLchar *unrealistic_fs =
    "#version 410\n"
    "in vec3 color;\n"
    "in float norm;\n"
    "layout (location = 0) out vec4 outColor;\n"
    "void main()\n"
    "{\n"
    "    outColor = vec4(color*max(vec3(norm),0.5)+.25,1.0f);\n"
    "}\n";

void prettyCoordinateAxes(
    const float *_camera, Vector3 position = Vector3(50, 50, 0),
    float arrow_size = 50, STYLE render_style = STYLE::LINE,
    bool render_text = false, Vector3 color_axis_x = Vector3(1, 0, 0),
    Vector3 color_axis_y = Vector3(0, 1, 0),
    Vector3 color_axis_z = Vector3(0, 0, 1), const char *axis_x = "X",
    const char *axis_y = "Y", const char *axis_z = "Z",
    Vector3 color_text_x = Vector3(1.0f), Vector3 color_text_y = Vector3(1.0f),
    Vector3 color_text_z = Vector3(1.0f))

{
  // Texture / shader program / vba
  GLint previous_active_texture;
  glGetIntegerv(GL_ACTIVE_TEXTURE, &previous_active_texture);
  glActiveTexture(GL_TEXTURE0);
  GLint previous_program;
  glGetIntegerv(GL_CURRENT_PROGRAM, &previous_program);
  GLint previous_texture;
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous_texture);
  GLint previous_array_buffer;
  glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &previous_array_buffer);
  GLint previous_vertex_array_object;
  glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &previous_vertex_array_object);
  // Blendings may change now and then. better keep them.
  GLint previous_blend_src_rgb;
  glGetIntegerv(GL_BLEND_SRC_RGB, &previous_blend_src_rgb);
  GLint previous_blend_dst_rgb;
  glGetIntegerv(GL_BLEND_DST_RGB, &previous_blend_dst_rgb);
  GLint previous_blend_src_alpha;
  glGetIntegerv(GL_BLEND_SRC_ALPHA, &previous_blend_src_alpha);
  GLint previous_blend_dst_alpha;
  glGetIntegerv(GL_BLEND_DST_ALPHA, &previous_blend_dst_alpha);
  GLint previous_blend_equation_rgb;
  glGetIntegerv(GL_BLEND_EQUATION_RGB, &previous_blend_equation_rgb);
  GLint previous_blend_equation_alpha;
  glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &previous_blend_equation_alpha);
  // States
  GLboolean previous_enable_blend = glIsEnabled(GL_BLEND);
  GLboolean previous_enable_cull_face = glIsEnabled(GL_CULL_FACE);
  GLboolean previous_enable_depth_test = glIsEnabled(GL_DEPTH_TEST);
  GLboolean previous_enable_scissor_test = glIsEnabled(GL_SCISSOR_TEST);

  auto restorePreviousState = [&]() -> void {
    // Destroy the Pretty VAO
    glDeleteVertexArrays(1, &prettyGeoVao);

    // Restore modified GL state
    glUseProgram(previous_program);
    glBindTexture(GL_TEXTURE_2D, previous_texture);
    glActiveTexture(previous_active_texture);
    glBindVertexArray(previous_vertex_array_object);
    glBindBuffer(GL_ARRAY_BUFFER, previous_array_buffer);
    glBlendEquationSeparate(previous_blend_equation_rgb,
                            previous_blend_equation_alpha);
    glBlendFuncSeparate(previous_blend_src_rgb, previous_blend_dst_rgb,
                        previous_blend_src_alpha, previous_blend_dst_alpha);
    if (previous_enable_blend)
      glEnable(GL_BLEND);
    else
      glDisable(GL_BLEND);
    if (previous_enable_cull_face)
      glEnable(GL_CULL_FACE);
    else
      glDisable(GL_CULL_FACE);
    if (previous_enable_depth_test)
      glEnable(GL_DEPTH_TEST);
    else
      glDisable(GL_DEPTH_TEST);
    if (previous_enable_scissor_test)
      glEnable(GL_SCISSOR_TEST);
    else
      glDisable(GL_SCISSOR_TEST);
  };
  // check for init
  if (!prettyHandleGeoProgram) {
    initGeometryShaderProgram();
#ifdef PRETTY_TEXT_RENDER
    initTextShaderProgram();
#endif
  }

  // I did not want to write a 2D vector just for position on screen.
  // Ensure that Vector3.z is 0
  if (!(position.z == 0))
    throw std::invalid_argument("Screen position cannot have a Z position");

  // Sector count for cylinders and cones incase needed.
  int sectorCount = 20;

  // Initiate Vertice and Indice arrays.
  std::vector<PrettyVrtx> vertices;
  std::vector<unsigned int> indices;

  Matrix4 camera(_camera);
  Vector3 origin = position;

  setOrthographicProjectionMatrix();

  // Currently just let the principal directions.
  std::vector<Vector3> axes = {Vector3(1, 0, 0), Vector3(0, 1, 0),
                               Vector3(0, 0, 1)};

  vertices.resize(6, PrettyVrtx(origin, Vector3(0, 0, 0)));

  // Set indices for lines
  indices.resize(6);
  std::iota(indices.begin(), indices.end(), 0);

  // no translation
  camera.resetTranslation();
  // inverse
  camera.inverted();

  for (auto i = 0; i < 3; i++)
    vertices[2 * i + 1].pos =
        origin + (camera * Vector4(axes[i] * arrow_size, 1)).xyz();

  vertices[0].col = vertices[1].col = color_axis_x;
  vertices[2].col = vertices[3].col = color_axis_y;
  vertices[4].col = vertices[5].col = color_axis_z;
#ifdef PRETTY_TEXT_RENDER

  if (render_text) {
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // Render text
    renderText(axis_x, .5, vertices[1].pos, color_text_x);
    renderText(axis_y, .5, vertices[3].pos, color_text_y);
    renderText(axis_z, .5, vertices[5].pos, color_text_z);
  }
#endif
  if (render_style == STYLE::LINE || render_style == STYLE::LINE_WITH_HEAD)
    renderGeometry(vertices, indices);
  if (render_style == STYLE::LINE) {
    return restorePreviousState();
  }
  // Generate Cylinder
  auto generateCylinder = [&](Vector3 axis, float degrees, Vector3 color,
                              float length, bool cone = false,
                              bool reverseCone = false) {
    vertices.clear();

    float radius = 1;
    if (cone)
      radius = 4;

    float sectorStep = 2 * PrettyXYZ::MathUtils::pi / sectorCount;

    Matrix4 rotationMat = MathUtils::rotate(degrees, axis);

    Vector3 t(0, 0, 4 * length);

    std::vector<PrettyVrtx> unitVertices;
    unitVertices.resize(sectorCount + 1);
    for (int i = 0; i <= sectorCount; ++i)
      unitVertices[i] = PrettyVrtx(
          Vector3(std::cos(i * sectorStep), sin(i * sectorStep), 0), color);
    // put side vertices to arrays
    for (int i = 0; i < 2; ++i) {
      float h = .0f + i * length;
      if (cone && i == 1)
        radius = 0;
      else if (reverseCone && i == 0)
        radius = 0;
      else if (reverseCone && i == 1)
        radius = 1;

      for (int j = 0, k = 0; j <= sectorCount; ++j, k++) {
        auto pos =
            unitVertices[k].pos + Vector3(unitVertices[k].pos.x * radius,
                                          unitVertices[k].pos.y * radius, h);
        pos = (rotationMat * Vector4(pos, 1.0)).xyz();
        auto normal = (rotationMat * Vector4(unitVertices[k].pos, 1)).xyz();

        if (cone) {

          pos += (rotationMat * Vector4(t, 1.0)).xyz();
        }

        pos = origin + (camera * Vector4(pos, 1)).xyz();

        vertices.push_back(PrettyVrtx(pos, normal, color));
      }
    }
    // put base and top vertices to arrays
    for (int i = 0; i < 2; ++i) {
      float h = 0 + i * length;
      // center point
      vertices.push_back(PrettyVrtx(Vector3(0, 0, h), color));
    }
  };
  updateIndicesForCylinder(indices, sectorCount);
  // Set OpenGL options
  glEnable(GL_CULL_FACE);
  bool _cone = false;
  bool _rev_cone = false;
  if (render_style == STYLE::LINE_WITH_HEAD ||
      render_style == STYLE::CYLINDER_WITH_HEAD ||
      render_style == STYLE::CONE) {
    _cone = true;
  }
  if (render_style != STYLE::CYLINDER) {
    // Heads
    generateCylinder(Vector3(0, 0, 1), 0, color_axis_z, arrow_size * 0.2f,
                     _cone);
    renderGeometry(vertices, indices);
    // X
    generateCylinder(Vector3(0, 1, 0), 90, color_axis_x, arrow_size * 0.2f,
                     _cone);
    renderGeometry(vertices, indices);
    // Y
    generateCylinder(Vector3(1, 0, 0), -90, color_axis_y, arrow_size * 0.2f,
                     _cone);
    renderGeometry(vertices, indices);
  }
  if (render_style == STYLE::LINE_WITH_HEAD)
    return restorePreviousState();
  else if (render_style == STYLE::CYLINDER_WITH_HEAD)
    _cone = false;
  else if (render_style == STYLE::CONE) {
    _cone = false;
    _rev_cone = true;
  }
  // Bases
  generateCylinder(Vector3(0, 0, 1), 0, color_axis_z, arrow_size * 0.8f, _cone,
                   _rev_cone);
  renderGeometry(vertices, indices);
  // X
  generateCylinder(Vector3(0, 1, 0), 90, color_axis_x, arrow_size * 0.8f, _cone,
                   _rev_cone);
  renderGeometry(vertices, indices);
  // Y
  generateCylinder(Vector3(1, 0, 0), -90, color_axis_y, arrow_size * 0.8f, _cone,
                   _rev_cone);
  renderGeometry(vertices, indices);
}
void updateIndicesForCylinder(std::vector<unsigned int> &indices,
                              int sectorCount) {
  indices.resize(12 * sectorCount);
  int c = 0;
  int k1 = 0;               // 1st vertex index at base
  int k2 = sectorCount + 1; // 1st vertex index at top

  // indices for the side surface
  for (int i = 0; i < sectorCount; ++i, ++k1, ++k2) {
    // 2 triangles per sector
    // k1 => k1+1 => k2
    indices[c++] = k1;
    indices[c++] = k1 + 1;
    indices[c++] = k2;

    // k2 => k1+1 => k2+1
    indices[c++] = k2;
    indices[c++] = k1 + 1;
    indices[c++] = k2 + 1;
  }

  // indices for the base surface
  for (int i = 0, k = 1; i < sectorCount; ++i, ++k) {
    if (i < sectorCount - 1) {
      indices[c++] = 0;
      indices[c++] = k + 1;
      indices[c++] = k;
    } else // last triangle
    {
      indices[c++] = 0;
      indices[c++] = 1;
      indices[c++] = k;
    }
  }

  // indices for the top surface
  for (int i = 0, k = sectorCount + 2; i < sectorCount; ++i, ++k) {
    if (i < sectorCount - 1) {
      indices[c++] = sectorCount + 1;
      indices[c++] = k;
      indices[c++] = k + 1;
    } else // last triangle
    {
      indices[c++] = sectorCount + 1;
      indices[c++] = k;
      indices[c++] = sectorCount + 2;
    }
  }
}

void initGeometryShaderProgram() {
  // Vertex shader
  GLuint vHandle = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vHandle, 1, &shaded_vs, nullptr);
  glCompileShader(vHandle);

  // Fragment shader
  GLuint fHandle = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fHandle, 1, &shaded_fs, nullptr);
  glCompileShader(fHandle);

  ShaderProgramCheck::checkShader(vHandle);
  ShaderProgramCheck::checkShader(fHandle);

  // Create Program
  prettyHandleGeoProgram = glCreateProgram();
  glAttachShader(prettyHandleGeoProgram, vHandle);
  glAttachShader(prettyHandleGeoProgram, fHandle);
  glLinkProgram(prettyHandleGeoProgram);
  ShaderProgramCheck::checkProgram(prettyHandleGeoProgram);
}

void renderGeometry(const std::vector<PrettyVrtx> &vertices,
                    std::vector<unsigned int> &indices) {
  glUseProgram(prettyHandleGeoProgram);
  glGenVertexArrays(1, &prettyGeoVao);

  // CreateBuffers
  glGenBuffers(1, &prettyGeoVbo);
  glGenBuffers(1, &prettyGeoIbo);

  glUniformMatrix4fv(glGetUniformLocation(prettyHandleGeoProgram, "projection"),
                     1, GL_FALSE, prettyOrthoProjection.begin());

  glBindVertexArray(prettyGeoVao);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prettyGeoIbo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int),
               indices.data(), GL_DYNAMIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, prettyGeoVbo);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(PrettyVrtx),
               vertices.data(), GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PrettyVrtx), nullptr);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(PrettyVrtx),
                        (void *)offsetof(PrettyVrtx, normal));
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_TRUE, sizeof(PrettyVrtx),
                        (void *)offsetof(PrettyVrtx, col));

  if (indices.size() == 6)
    glDrawElements(GL_LINES, (GLsizei)indices.size(), GL_UNSIGNED_INT, nullptr);
  else
    glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT,
                   nullptr);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  glUseProgram(0);
}

void setOrthographicProjectionMatrix() {
  GLint previous_viewport[4];
  glGetIntegerv(GL_VIEWPORT, previous_viewport);
  float L = static_cast<float>(previous_viewport[0]);
  float R = L + static_cast<float>(previous_viewport[2]);

  // B-T ? or T-B check. TODO
  float B = static_cast<float>(previous_viewport[1]);
  float T = B + static_cast<float>(previous_viewport[3]);
  float N = -128.0;
  float F = 128.0;

  Matrix4 orthoProjection(0.0f);
  orthoProjection[0] = 2.0f / (R - L);
  orthoProjection[5] = 2.0f / (T - B);
  orthoProjection[10] = -2 / (F - N);
  orthoProjection[12] = (R + L) / (L - R);
  orthoProjection[13] = (T + B) / (B - T);
  orthoProjection[15] = 1.0f;

  prettyOrthoProjection = orthoProjection;
}

#ifdef PRETTY_TEXT_RENDER
void initTextShaderProgram() {
  // Vertex shader
  GLuint vHandle = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vHandle, 1, &text_vs, nullptr);
  glCompileShader(vHandle);
  // Geometry shader
  GLuint gHandle = glCreateShader(GL_GEOMETRY_SHADER);
  glShaderSource(gHandle, 1, &text_gs, nullptr);
  glCompileShader(gHandle);
  // Fragment shader
  GLuint fHandle = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fHandle, 1, &text_fs, nullptr);
  glCompileShader(fHandle);

  ShaderProgramCheck::checkShader(vHandle);
  ShaderProgramCheck::checkShader(gHandle);
  ShaderProgramCheck::checkShader(fHandle);

  // Create Program
  prettyHandleTextProgram = glCreateProgram();
  glAttachShader(prettyHandleTextProgram, vHandle);
  glAttachShader(prettyHandleTextProgram, gHandle);
  glAttachShader(prettyHandleTextProgram, fHandle);
  glLinkProgram(prettyHandleTextProgram);
  ShaderProgramCheck::checkProgram(prettyHandleTextProgram);

  // Init freetype
  // FreeType
  // --------
  FT_Library ft;
  // All functions return a value different than 0 whenever an error occurred
  if (FT_Init_FreeType(&ft))
    std::cout << "ERROR::FREETYPE: Could not init FreeType Library"
              << std::endl;

  // load font as face
  FT_Face face;
  if (FT_New_Face(ft, "fonts/StarellaTattoo.ttf", 0, &face))
    std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;

  // set size to load glyphs as
  FT_Set_Pixel_Sizes(face, 0, 48);

  // disable byte-alignment restriction
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  // load first 128 characters of ASCII set
  for (unsigned char c = 0; c < 128; c++) {
    // Load character glyph
    if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
      std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
      continue;
    }
    // generate texture
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, face->glyph->bitmap.width,
                 face->glyph->bitmap.rows, 0, GL_RED, GL_UNSIGNED_BYTE,
                 face->glyph->bitmap.buffer);
    // set texture options
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // now store character for later use
    Character character = {
        texture,
        {face->glyph->bitmap.width, face->glyph->bitmap.rows},
        face->glyph->advance.x};
    Characters.insert(std::pair<char, Character>(c, character));
  }
  glBindTexture(GL_TEXTURE_2D, 0);
  // destroy FreeType once we're finished
  FT_Done_Face(face);
  FT_Done_FreeType(ft);
  glGenVertexArrays(1, &prettyTextVao);
  glGenBuffers(1, &prettyTextVbo);
  glBindVertexArray(prettyTextVao);
  glBindBuffer(GL_ARRAY_BUFFER, prettyTextVbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, nullptr,
               GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}
void renderText(std::string text, float scale, Vector3 location,
                Vector3 color) {
  // activate corresponding render state
  glUseProgram(prettyHandleTextProgram);
  glUniform3fv(glGetUniformLocation(prettyHandleTextProgram, "textColor"), 1,
               &(color.x));
  glUniformMatrix4fv(
      glGetUniformLocation(prettyHandleTextProgram, "projection"), 1, GL_FALSE,
      prettyOrthoProjection.begin());

  glActiveTexture(GL_TEXTURE0);
  glBindVertexArray(prettyTextVao);

  // iterate through all characters
  std::string::const_iterator c;
  for (c = text.begin(); c != text.end(); c++) {
    Character ch = Characters[*c];

    glUniform1f(glGetUniformLocation(prettyHandleTextProgram, "width"),
                ch.Size[0] * scale);
    glUniform1f(glGetUniformLocation(prettyHandleTextProgram, "height"),
                ch.Size[1] * scale);

    // render glyph texture over quad
    glBindTexture(GL_TEXTURE_2D, ch.TextureID);
    // update content of VBO memory
    glBindBuffer(GL_ARRAY_BUFFER, prettyTextVbo);
    glBufferSubData(
        GL_ARRAY_BUFFER, 0, 3 * sizeof(float),
        &(location.x)); // be sure to use glBufferSubData and not glBufferData

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // render quad
    glDrawArrays(GL_POINTS, 0, 1);
    // now advance cursors for next glyph (note that advance is number of 1/64
    // pixels)
    location.x +=
        (ch.Advance >> 6) *
        scale; // bitshift by 6 to get value in pixels (2^6 = 64 (divide amount
               // of 1/64th pixels by 64 to get amount of pixels))
  }
  glBindVertexArray(0);
  glBindTexture(GL_TEXTURE_2D, 0);
}
#endif

namespace ShaderProgramCheck {
void checkShader(GLuint shaderHandle) {
  GLint _result = 0;
  GLchar _eCode[1024] = {0};

  glGetShaderiv(shaderHandle, GL_COMPILE_STATUS, &_result);
  if (!_result) {
    glGetShaderInfoLog(shaderHandle, sizeof(_eCode), nullptr, _eCode);
    std::cout << "Error happened while compuling vertex shader" << _eCode
              << std::endl;
    return;
  }
  std::cout << "ShaderCreated" << std::endl;
}
void checkProgram(GLuint programID) {
  GLint _result = 0;
  GLchar _eCode[1024] = {0};
  glGetProgramiv(programID, GL_LINK_STATUS, &_result);
  if (!_result) {
    glGetProgramInfoLog(programID, sizeof(_eCode), nullptr, _eCode);
    std::cout << "Error happened while linking program " << _eCode << std::endl;
    throw;
  }

  glValidateProgram(programID);

  glGetProgramiv(programID, GL_LINK_STATUS, &_result);
  if (!_result) {
    glGetProgramInfoLog(programID, sizeof(_eCode), nullptr, _eCode);
    std::cout << "Error happened while validating program " << _eCode
              << std::endl;
    throw;
  }
  std::cout << "Shader program is created without a problem" << std::endl;
}
} // namespace ShaderProgramCheck

} // namespace PrettyXYZ
