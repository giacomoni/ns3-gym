/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Technische Universität Berlin
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Piotr Gawlowicz <gawlowicz@tkn.tu-berlin.de>
 */

#include "cartpolegym.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/wifi-module.h"
#include "ns3/node-list.h"
#include "ns3/log.h"
#include <math.h>
#include <sstream>
#include <iostream>
#include <string>
#include <random>
#include <limits>


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("CartpoleGymEnv");

NS_OBJECT_ENSURE_REGISTERED (CartpoleGymEnv);

CartpoleGymEnv::CartpoleGymEnv ()
{
  NS_LOG_FUNCTION (this);
  m_interval = Seconds(1);
  initialise();

  Simulator::Schedule (Seconds(0.0), &CartpoleGymEnv::ScheduleNextStep, this);
}

CartpoleGymEnv::CartpoleGymEnv (Time stepTime)
{
  NS_LOG_FUNCTION (this);
  m_interval = stepTime;
  initialise();

  Simulator::Schedule (Seconds(0.0), &CartpoleGymEnv::ScheduleNextStep, this);
}

void
CartpoleGymEnv::initialise(){
    gravity = 9.8;
    masscart = 1.0;
    masspole = 0.1;
    total_mass = masspole + masscart;
    length = 0.5; // actually half the pole's length
    polemass_length = masspole * length;
    force_mag = 10.0;
    tau = 0.02; // seconds between state updates
    kinematics_integrator = "euler";

    // Angle at which to fail the episode
    theta_threshold_radians = 12 * 2 * M_PI / 360;
    x_threshold = 2.4;

    // Angle limit set to 2 * theta_threshold_radians so failing observation
    // is still within bounds.
    high[0] = x_threshold * 2;
    high[1] = 3.4028235e+38;
    high[2] = theta_threshold_radians * 2;
    high[3] = 3.4028235e+38;

    a = 2;
    b = 13;    

    randomState(state);
}


// Initial state of the cartpole is random
void
CartpoleGymEnv::randomState(float *state)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.05, 0.05);
    for (int n = 0; n < 4; ++n)
        state[n] = dis(gen);
}


void
CartpoleGymEnv::ScheduleNextStep ()
{
  NS_LOG_FUNCTION (this);
  Simulator::Schedule (m_interval, &CartpoleGymEnv::ScheduleNextStep, this);
  Notify();
}

CartpoleGymEnv::~CartpoleGymEnv ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
CartpoleGymEnv::GetTypeId (void)
{
  static TypeId tid = TypeId ("CartpoleGymEnv")
    .SetParent<OpenGymEnv> ()
    .SetGroupName ("OpenGym")
    .AddConstructor<CartpoleGymEnv> ()
  ;
  return tid;
}

void
CartpoleGymEnv::DoDispose ()
{
  NS_LOG_FUNCTION (this);
}

/*
Define observation space
*/
Ptr<OpenGymSpace>
CartpoleGymEnv::GetObservationSpace()
{
  uint32_t nodeNum = 4;
  float low = 0.0;
  float high = 10.0;
  std::vector<uint32_t> shape = {nodeNum,};
  std::string dtype = TypeNameGet<float> ();

  Ptr<OpenGymBoxSpace> box = CreateObject<OpenGymBoxSpace> (-std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), shape, dtype);

  NS_LOG_UNCOND ("ObservationSpace: " << box);
  return box;
}

/*
Define action space
*/
Ptr<OpenGymSpace>
CartpoleGymEnv::GetActionSpace()
{
  uint32_t nodeNum = 2;
  std::vector<uint32_t> shape = {nodeNum,};

  Ptr<OpenGymDiscreteSpace> discrete = CreateObject<OpenGymDiscreteSpace> (nodeNum);

  NS_LOG_UNCOND ("ActionSpace: " << discrete);
  return discrete;
}

/*
Define game over condition
*/
bool
CartpoleGymEnv::GetGameOver()
{
  bool isGameOver = false;
  static float stepCounter = 0.0;
  stepCounter += 1;

  if (state[0] < x_threshold * -1 || state[0] > x_threshold || state[2] < theta_threshold_radians * -1 || state[2] > theta_threshold_radians || stepCounter >= 400)
    {
        isGameOver = true;
    }
  
  NS_LOG_UNCOND ("MyGetGameOver: " << isGameOver);
  return isGameOver;
}

/*
Collect observations
*/
Ptr<OpenGymDataContainer>
CartpoleGymEnv::GetObservation()
{
  uint32_t nodeNum = 4;
  std::vector<uint32_t> shape = {nodeNum,};
  Ptr<OpenGymBoxContainer<float> > box = CreateObject<OpenGymBoxContainer<float> >(shape);

  // Copy state values from state array
  for (uint32_t i = 0; i<nodeNum; i++){
    float value = state[i];
    box->AddValue(value);
  }

  // Print data from tuple
  Ptr<OpenGymBoxContainer<float> > mbox = DynamicCast<OpenGymBoxContainer<float> >(box);
  NS_LOG_UNCOND ("---" << mbox);
  return mbox;
}

/*
Define reward function
*/
float
CartpoleGymEnv::GetReward()
{
  static float reward = 1.0;
  return reward;
}

/*
Define extra info. Optional
*/
std::string
CartpoleGymEnv::GetExtraInfo()
{
  std::string myInfo = "testInfo";
  myInfo += "|123";
  NS_LOG_UNCOND("MyGetExtraInfo: " << myInfo);
  return myInfo;
}

/*
Execute received actions
*/
bool
CartpoleGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
    Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);

    auto action_t = discrete->GetValue();
    float x = state[0];
    float x_dot = state[1];
    float theta = state[2];
    float theta_dot = state[3];
    float force;

    if (action_t == 1)
    {
        force = force_mag;
    }
    else
    {
        force = force_mag * -1;
    }

    float costheta = cos(theta);
    float sintheta = sin(theta);

    float temp = (force + polemass_length * pow(theta_dot, 2) * sintheta) / total_mass;

    float thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * pow(costheta, 2) / total_mass));

    float xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    // cout << "temp: " << temp << " thetaacc: " << thetaacc << " xacc: " << xacc <<endl;

    if (kinematics_integrator == "euler")
    {
        x = x + tau * x_dot;
        x_dot = x_dot + tau * xacc;
        theta = theta + tau * theta_dot;
        theta_dot = theta_dot + tau * thetaacc;
    }
    else
    {
        x_dot = x_dot + tau * xacc;
        x = x + tau * x_dot;
        theta_dot = theta_dot + tau * thetaacc;
        theta = theta + tau * theta_dot;
    }

    state[0] = x;
    state[1] = x_dot;
    state[2] = theta;
    state[3] = theta_dot;


  NS_LOG_UNCOND ("MyExecuteActions: " << action);
  NS_LOG_UNCOND ("---" << discrete);
  return true;
}

} // ns3 namespace