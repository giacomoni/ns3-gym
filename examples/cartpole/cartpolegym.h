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


#ifndef CARTPOLE_GYM_ENTITY_H
#define CARTPOLE_GYM_ENTITY_H

#include "ns3/opengym-module.h"
#include "ns3/nstime.h"




namespace ns3 {

class CartpoleGymEnv : public OpenGymEnv
{
public:
  CartpoleGymEnv ();
  CartpoleGymEnv (Time stepTime);
  virtual ~CartpoleGymEnv ();
  static TypeId GetTypeId (void);
  virtual void DoDispose ();

  Ptr<OpenGymSpace> GetActionSpace();
  Ptr<OpenGymSpace> GetObservationSpace();
  bool GetGameOver();
  Ptr<OpenGymDataContainer> GetObservation();
  float GetReward();
  std::string GetExtraInfo();
  bool ExecuteActions(Ptr<OpenGymDataContainer> action);

private:
  float a;
  float b;
  float gravity;
  float masscart;
  float masspole;
  float total_mass;
  float length; // actually half the pole's length
  float polemass_length;
  float force_mag;
  float tau; // seconds between state updates
  std::string kinematics_integrator;

  float theta_threshold_radians;
  float x_threshold;

  float high[4];
  float state[4];

  
  Time m_interval;

  void ScheduleNextStep();
  void initialise();
  void randomState(float *state);


};

}


#endif // CARTPOLE_GYM_ENTITY_H
