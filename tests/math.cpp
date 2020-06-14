#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include "3rdparty/catch.hpp"
#define PRETTY_TEXT_RENDER
#include "../PrettyXYZ.hpp"

using namespace PrettyXYZ::MathUtils;

TEST_CASE("Matrix rotation", "Rotation")
{

    Matrix4 expected({0.7268376, 0.6414292, -0.2455112, 0,
                      -0.4733292, 0.7268376, 0.4976611, 0,
                      0.4976611, -0.2455112, 0.8319001, 0,
                      0, 0, 0, 1});

    Vector3 axis(2, 2, 3);
    axis.normalized();
    Matrix4 real = rotate(50, axis);
    for (int i = 0; i < 16; i++)
        REQUIRE(real[i] == Approx(expected[i])); // Approx is needed due to floating point comparison
}

TEST_CASE("Math operation", "MathOp")
{
    Vector3 V1(3);
    Vector3 V2(5);
    Vector3 V3 = V1 + V2;
    REQUIRE(V3.x == 8.0f);
    REQUIRE(V3.y == 8.0f);
    REQUIRE(V3.z == 8.0f);

    Matrix4 M(10);
    Vector4 V4 = M * Vector4(V1, 3);
    REQUIRE(V4.x == 120.0f);
    REQUIRE(V4.y == 120.0f);
    REQUIRE(V4.z == 120.0f);
    REQUIRE(V4.w == 120.0f);
    REQUIRE(V4.xyz().x == 120.0f);
    REQUIRE(V4.xyz().y == 120.0f);
    REQUIRE(V4.xyz().z == 120.0f);

    Matrix4 M2(15);
    Matrix4 M3 = M * M2;
    for (int i = 0; i < 16; i++)
        REQUIRE(M3[i] == 600.0f);

    Matrix4 M4(1.0f);
    Matrix4 M5(1.0f);
    M5[1] = M5[2] = M5[3] = M5[6] = M5[7] = M5[11] = -1.0f;
    Matrix4 M6 = M5 * M4;
    for (int i = 0; i < 16; i++)
    {
        switch (i % 4)
        {
        case 0:
            REQUIRE(M6[i] == 4);
            break;
        case 1:
            REQUIRE(M6[i] == 2);
            break;
        case 2:
            REQUIRE(M6[i] == 0);
            break;
        case 3:
            REQUIRE(M6[i] == -2);
            break;
        default:
            throw;
        }
    }
}
TEST_CASE("Vec3 Copy constructor", "vec3copy")
{
    Vector3 V1(3);
    Vector3 V2(5);
    V1 = V2;
    REQUIRE(V1.x == 5.0f);
    REQUIRE(V1.y == 5.0f);
    REQUIRE(V1.z == 5.0f);
}
TEST_CASE("MatrixFunctions", "mat4funcs")
{
    Matrix4 M(4.0f);
    M.resetTranslation();
    REQUIRE(M[12] == 0.0f);
    REQUIRE(M[13] == 0.0f);
    REQUIRE(M[14] == 0.0f);

    float example_matrix[16] = {1, 0, 2, 1, 1, 3, 3, 0, 1, 1, 1, 2, 0, 2, 0, 1};
    float inverted_matrix[16] = {-3, 1, 3, -3, -0.5, 0.25, 0.25, -0, 1.5, -0.25, -1.25, 1, 1, -0.5, -0.5, 1};

    Matrix4 M2(example_matrix);
    M2.inverted();
    for (int i = 0; i < 16; i++)
        REQUIRE(M2[i] == inverted_matrix[i]);
}
TEST_CASE("Pointer", "[ptr]")
{
    Vector3 V(3, 5, 7.5);
    REQUIRE(static_cast<void *>(&(V)) == static_cast<void *>(&(V.x)));

    Matrix4 M(3);
    REQUIRE(reinterpret_cast<float *>(&(M)) == M.begin());
}
TEST_CASE("Matrix4 constructor", "[matrix44const]")
{

    Matrix4 C0(3);
    for (int i = 0; i < 16; i++)
        REQUIRE(C0[i] == 3.0f);

    Matrix4 C1(C0);
    for (int i = 0; i < 16; i++)
        REQUIRE(C1[i] == 3.0f);

    Matrix4 C2(C1);
    for (int i = 0; i < 16; i++)
        REQUIRE(C2[i] == 3.0f);

    C0[3] = 26.0f;
    C0[10] = 0.0005f;

    Matrix4 C4(4);
    C4 = C0;
    REQUIRE(C4[3] == 26.0f);
    REQUIRE(C4[10] == 0.0005f);
}
TEST_CASE("Vector4 constructor", "[vector4const]")
{

    Vector4 C0;
    REQUIRE(C0.x == 0.0f);
    REQUIRE(C0.y == 0.0f);
    REQUIRE(C0.z == 0.0f);
    REQUIRE(C0.w == 0.0f);

    Vector4 C1(5.2);
    REQUIRE(C1.x == 5.2f);
    REQUIRE(C1.y == 5.2f);
    REQUIRE(C1.z == 5.2f);
    REQUIRE(C1.w == 5.2f);

    Vector4 C2(3, 5, 7.5, -3);
    REQUIRE(C2.x == 3.0f);
    REQUIRE(C2.y == 5.0f);
    REQUIRE(C2.z == 7.5f);
    REQUIRE(C2.w == -3.0f);

    Vector4 C3(Vector3(3, 5, 7), -3);
    REQUIRE(C3.x == 3.0f);
    REQUIRE(C3.y == 5.0f);
    REQUIRE(C3.z == 7.0f);
    REQUIRE(C3.w == -3.0f);
}
TEST_CASE("Vector3 constructor", "[vector3const]")
{

    Vector3 C0;
    REQUIRE(C0.x == 0.0f);
    REQUIRE(C0.y == 0.0f);
    REQUIRE(C0.z == 0.0f);

    Vector3 C1(5.2);
    REQUIRE(C1.x == 5.2f);
    REQUIRE(C1.y == 5.2f);
    REQUIRE(C1.z == 5.2f);

    Vector3 C2(3, 5, 7.5);
    REQUIRE(C2.x == 3.0f);
    REQUIRE(C2.y == 5.0f);
    REQUIRE(C2.z == 7.5f);
}
