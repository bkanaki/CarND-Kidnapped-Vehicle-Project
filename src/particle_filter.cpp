/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 50;

	// random generator
	default_random_engine gen;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// initialize all the particles with random orientation and position
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
		weights.push_back(p.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	for (int i = 0; i < num_particles; i++) {
		double new_x;
		double new_y;
		double new_theta;

		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		if (fabs(yaw_rate) < 1e-5) {
			// based on Taylor series expansion of original formula in else block
			new_x = p_x + velocity * delta_t * cos(p_theta);
			new_y = p_y + velocity * delta_t * sin(p_theta);
			new_theta = p_theta;
		} else {
			new_x = p_x + (velocity / yaw_rate) * 
										(sin(p_theta + yaw_rate * delta_t) - sin(p_theta));
			new_y = p_y + (velocity / yaw_rate) * 
										(cos(p_theta) - cos(p_theta + yaw_rate * delta_t));
			new_theta = p_theta + yaw_rate * delta_t;
		}

		normal_distribution<double> dist_x(new_x, std_pos[0]);
		normal_distribution<double> dist_y(new_y, std_pos[1]);
		normal_distribution<double> dist_theta(new_theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// do the nearest neighbour association
	for (auto &observation : observations) {
		// set the minimum distance to a very large value
		double min_dist = 1e20;

		// set the match id to -1 for each observation to begin with
		int match_id = -1;

		for (const auto &prediction : predicted) {
			double dist_op = dist(observation.x, observation.y, 
														prediction.x, prediction.y);
			
			// update the min distance and match id if this prediction is closer
			if (dist_op < min_dist) {
				min_dist = dist_op;
				match_id = prediction.id;
			}
		}

		// finally, update the observation id with the closest prediction id
		observation.id = match_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// to be used later for the update of weights
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
	double gauss_norm = 1 / (2 * M_PI * std_x * std_y);

	// for each particle in particles
	for (auto &particle : particles) {
		double p_x = particle.x;
		double p_y = particle.y;
		double p_theta = particle.theta;

		// list of landmark locations that are within the sensor range
		vector<LandmarkObs> predictions;

		for (const auto &landmark : map_landmarks.landmark_list) {
			double lm_x = landmark.x_f;
			double lm_y = landmark.y_f;
			int lm_id = landmark.id_i;

			// add only if it is within the range from particle
			// double dist = sqrt((p_x - lm_x) * (p_x - lm_x) + 
			// 									 (p_y - lm_y) * (p_y - lm_y));
			if(dist(p_x, p_y, lm_x, lm_y) < sensor_range) {
				predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
			}
		}

		// transform the observations from the car coordinates to map coordinates
		vector<LandmarkObs> tx_observations;
		for (const auto &observation : observations) {
			double t_x = p_x + observation.x * cos(p_theta) - 
									 observation.y * sin(p_theta);
			double t_y = p_y + observation.x * sin(p_theta) + 
									 observation.y * cos(p_theta);
			tx_observations.push_back(LandmarkObs{observation.id, t_x, t_y});
		}

		// do the data association using knn
		dataAssociation(predictions, tx_observations);

		// reinitialize the weights
		particle.weight = 1.0;

		// to store associated prediction
		double ap_x, ap_y;

		// update the weight using multivariate Gaussian distribution
		for (const auto &tx_obs : tx_observations) {
			for (const auto &prediction : predictions) {
				if (tx_obs.id == prediction.id) {
					ap_x = prediction.x;
					ap_y = prediction.y;
				}
			}

			// calculate the weight 
			double obs_x = tx_obs.x;
			double obs_y = tx_obs.y;

			double exponent = pow(ap_x - obs_x, 2) / (2 * pow(std_x, 2)) +
												pow(ap_y - obs_y, 2) / (2 * pow(std_y, 2));
			
			particle.weight *= gauss_norm * exp(-exponent);
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	default_random_engine gen;

	vector<Particle> new_particles;

  // get all of the current weights
  vector<double> weights;
	for (const auto &particle : particles) {
		weights.push_back(particle.weight);
	}

  // generate random starting index for resampling wheel
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);

  // get max weight
  double w_max = *max_element(weights.begin(), weights.end());

  // uniform random distribution [0.0, w_max)
  uniform_real_distribution<double> unirealdist(0.0, w_max);

  double beta = 0.0;

  // resampling wheel
  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;	// mod is the trick!
    }
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
		return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
