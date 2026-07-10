#include <Eigen/Core>
#include <gtest/gtest.h>

TEST(EigenSmokeTest, ComputesVectorNorm) {
  const Eigen::Vector2d vector{3.0, 4.0};

  EXPECT_DOUBLE_EQ(vector.norm(), 5.0);
}
