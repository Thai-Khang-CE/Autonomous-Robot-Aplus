// Bounded force-driven prismatic obstacle, synchronized to simulation time.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components/JointForceCmd.hh>
#include <gz/sim/components/JointPosition.hh>
#include <gz/sim/components/JointVelocity.hh>
#include <memory>
#include <sdf/Element.hh>

namespace robot_aplus
{
class ObstaclePatrol : public gz::sim::System,
                       public gz::sim::ISystemConfigure,
                       public gz::sim::ISystemPreUpdate
{
public:
  void Configure(const gz::sim::Entity& entity, const std::shared_ptr<const sdf::Element>& sdf,
                 gz::sim::EntityComponentManager& ecm, gz::sim::EventManager&) override
  {
    joint_ = gz::sim::Model(entity).JointByName(ecm, "slide");
    amplitude_ = sdf->Get<double>("amplitude");
    period_ = sdf->Get<double>("period");
    if (joint_ == gz::sim::kNullEntity || amplitude_ <= 0 || period_ <= 0) {
      joint_ = gz::sim::kNullEntity;
      gzerr << "ObstaclePatrol: invalid slide joint or trajectory\n";
      return;
    }
    ecm.CreateComponent(joint_, gz::sim::components::JointPosition());
    ecm.CreateComponent(joint_, gz::sim::components::JointVelocity());
    ecm.CreateComponent(joint_, gz::sim::components::JointForceCmd({0.0}));
  }

  void PreUpdate(const gz::sim::UpdateInfo& info, gz::sim::EntityComponentManager& ecm) override
  {
    if (info.paused || joint_ == gz::sim::kNullEntity) return;
    auto* position = ecm.Component<gz::sim::components::JointPosition>(joint_);
    auto* velocity = ecm.Component<gz::sim::components::JointVelocity>(joint_);
    auto* force = ecm.Component<gz::sim::components::JointForceCmd>(joint_);
    if (!position || !velocity || !force || position->Data().empty() || velocity->Data().empty())
      return;
    const double t = std::chrono::duration<double>(info.simTime).count();
    const double frequency = 2.0 * std::acos(-1.0) / period_;
    const double target = amplitude_ * std::sin(frequency * t);
    const double target_velocity = amplitude_ * frequency * std::cos(frequency * t);
    // Finite force means contact can oppose motion; never teleport through a robot.
    const double command =
        40.0 * (target - position->Data()[0]) + 12.0 * (target_velocity - velocity->Data()[0]);
    force->Data()[0] = std::clamp(command, -20.0, 20.0);
  }

private:
  gz::sim::Entity joint_{gz::sim::kNullEntity};
  double amplitude_{1.0};
  double period_{32.0};
};
}  // namespace robot_aplus

GZ_ADD_PLUGIN(robot_aplus::ObstaclePatrol, gz::sim::System,
              robot_aplus::ObstaclePatrol::ISystemConfigure,
              robot_aplus::ObstaclePatrol::ISystemPreUpdate)
