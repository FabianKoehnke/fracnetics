#ifndef GYMNASIUM_WRAPPER_HPP
#define GYMNASIUM_WRAPPER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "Network.hpp"

namespace py = pybind11;
using namespace py::literals;

/**
 * @file GymnasiumWrapper.hpp
 * @brief Problem-independent evaluation of GNP networks in Gymnasium environments.
 *
 * @details
 * The only part of fracnetics that talks to Python. The core (Node, Network, Population)
 * is plain C++; this header runs the episode loop against any environment with the
 * Gymnasium API (reset(seed, options) -> (obs, info), step(action) ->
 * (obs, reward, terminated, truncated, info)) and writes the results into the networks.
 *
 * Everything problem-specific stays outside the library:
 * - reward shaping, curricula, modified start states: a gymnasium.Wrapper around the env,
 *   with per-episode settings passed through resetOptions;
 * - additional selection criteria: the episodeObjectives hook, called once per episode
 *   with an EpisodeResult and returning that episode's objective vector.
 *
 * See https://gymnasium.farama.org
 */

/**
 * @brief Everything known about one finished episode; handed to the episodeObjectives hook.
 *
 * @details Built only when a hook is set. The observations and the info dict are the
 * objects the environment returned, not copies.
 */
struct EpisodeResult {
    int seed = -1;                    /**< Seed passed to env.reset() (-1 = unseeded). */
    py::object initialObservation;    /**< Observation returned by env.reset(). */
    py::object observation;           /**< Observation after the last step. */
    py::object info;                  /**< Info dict of the last step. */
    py::array_t<float> rewards;       /**< Reward of every step, in order (float32, as summed into the return). */
    float episodeReturn = 0.0f;       /**< Undiscounted sum of rewards, accumulated in float like Network::fitness. */
    int steps = 0;                    /**< Number of env.step() calls. */
    bool terminated = false;          /**< terminated flag of the last step. */
    bool truncated = false;           /**< truncated flag of the last step (false if the episode hit maxSteps). */
    bool invalid = false;             /**< The network exceeded dMax at some point (and used fallbackAction from then on). */
};

/**
 * @brief Episode-level settings shared by every episode of an evaluation.
 */
struct EpisodeSettings {
    int dMax = 10;                    /**< Maximum consecutive judgment nodes per decision. */
    int maxSteps = 1000;              /**< Episode length limit, on top of the environment's own termination. */
    int maxConsecutiveP = 5;          /**< Maximum consecutive processing nodes before fallbackAction is used. */
    int fallbackAction = 0;           /**< Action executed for an invalid network (see Network::act()). */
    int holdAction = -1;              /**< Processing function that repeats the previous decision; -1 = off. */
};

/**
 * @brief Copies an observation into out, reusing its storage.
 *
 * @details 1-D float32/float64 numpy arrays are read through the buffer; the generic
 * sequence cast creates one Python float per element, which was most of the per-step
 * cost on the C++ side. The values are the same: float32 -> double is exact.
 * Anything else falls back to the sequence cast.
 *
 * @param src Observation as returned by reset()/step().
 * @param out Destination, resized to the observation length.
 */
inline void readObservation(const py::handle& src, std::vector<double>& out) {
    if (py::isinstance<py::array_t<float, py::array::c_style>>(src)) {
        auto arr = py::reinterpret_borrow<py::array_t<float, py::array::c_style>>(src);
        if (arr.ndim() == 1) {
            const float* p = arr.data();
            out.assign(p, p + arr.size());
            return;
        }
    } else if (py::isinstance<py::array_t<double, py::array::c_style>>(src)) {
        auto arr = py::reinterpret_borrow<py::array_t<double, py::array::c_style>>(src);
        if (arr.ndim() == 1) {
            const double* p = arr.data();
            out.assign(p, p + arr.size());
            return;
        }
    }
    out = src.cast<std::vector<double>>();
}

/**
 * @brief Runs one episode of a network in a Gymnasium environment.
 *
 * @details
 * 1. env.reset(seed=seed[, options=options]); seed < 0 resets without a seed.
 * 2. Network::initPathTraversal(0, newRun): with newRun = false the node usage flags and
 *    counters keep accumulating (several cases of one generation).
 * 3. Until terminated, truncated or maxSteps: Network::act() picks the action,
 *    env.step(action), Network::recordStep() adds the reward to fitness.
 * 4. Network::updateExperienceFromEpisode() (JE nodes).
 * 5. If objectivesHook is set, it is called with the EpisodeResult and its return value
 *    (a sequence of floats, larger = better) is returned.
 *
 * Afterwards network.fitness holds the undiscounted episode return.
 *
 * @param net Network to run.
 * @param env Gymnasium environment (or anything with the same reset/step signature).
 * @param seed Seed for env.reset(); negative = no seed.
 * @param options Passed as env.reset(options=...) unless None, e.g. per-episode settings for a wrapper.
 * @param newRun Reset usage flags and traverse counters before the episode.
 * @param settings Traversal limits and fallback/hold actions, see EpisodeSettings.
 * @param objectivesHook None, or callable(EpisodeResult) -> sequence of floats.
 * @return The hook's objective vector; empty without a hook.
 * @throws std::invalid_argument if the observation has fewer values than the network has
 *         judgment functions (jnf) -- the judgment nodes would read past its end.
 */
inline std::vector<float> runEpisode(Network& net, const py::object& env, int seed,
                                     const py::object& options, bool newRun,
                                     const EpisodeSettings& settings,
                                     const py::object& objectivesHook) {
    py::object reset = env.attr("reset");
    py::tuple resetOut;
    if (options.is_none()) {
        resetOut = (seed >= 0) ? reset("seed"_a = seed) : reset();
    } else {
        resetOut = (seed >= 0) ? reset("seed"_a = seed, "options"_a = options) : reset("options"_a = options);
    }
    std::vector<double> obs;
    readObservation(resetOut[0], obs);
    if (obs.size() < net.jnf) {
        throw std::invalid_argument("gymnasium(): the observation has " + std::to_string(obs.size()) +
                                    " values, but the network uses jnf=" + std::to_string(net.jnf) +
                                    " judgment functions (one per observation value)");
    }

    net.initPathTraversal(0, newRun);

    const bool withHook = !objectivesHook.is_none();
    std::vector<float> rewards;
    py::object lastObservation, lastInfo;
    bool terminated = false, truncated = false;
    int steps = 0;
    py::object step = env.attr("step"); // bound once instead of an attribute lookup per step

    while (true) {
        int action = net.act(obs, settings.dMax, settings.maxConsecutiveP,
                             settings.fallbackAction, settings.holdAction);
        py::tuple result = step(py::int_(action));
        readObservation(result[0], obs);
        const float reward = result[1].cast<float>();
        terminated = result[2].cast<bool>();
        truncated = result[3].cast<bool>();
        steps++;
        net.recordStep(reward);
        if (withHook) {
            rewards.push_back(reward);
            lastObservation = result[0];
            lastInfo = result[4];
        }
        if (terminated || truncated || steps >= settings.maxSteps) break;
    }

    net.updateExperienceFromEpisode();

    if (!withHook) return {};
    EpisodeResult episode;
    episode.seed = seed;
    episode.initialObservation = resetOut[0];
    episode.observation = lastObservation;
    episode.info = lastInfo;
    episode.rewards = py::array_t<float>(rewards.size(), rewards.data());
    episode.episodeReturn = net.fitness;
    episode.steps = steps;
    episode.terminated = terminated;
    episode.truncated = truncated;
    episode.invalid = net.invalid;
    return objectivesHook(episode).cast<std::vector<float>>();
}

/**
 * @brief Evaluates networks on a batch of seeds; one episode per seed is one case.
 *
 * @details
 * For every network and every seed one episode is run (runEpisode()). Per network:
 * - caseScores[c]: the return of the episode on seeds[c];
 * - visitedNodesPerCase[c]: the nodes entered during that episode;
 * - objectivesPerCase[c]: the hook's objectives (only with a hook);
 * - fitness / lexicaseObjectives: their means (Network::aggregateCaseResults()).
 * Node usage flags and counters accumulate over all seeds of the call, so they describe
 * the whole evaluation (callAddDelNodes(), mutation with justUsedNodes).
 *
 * @param individuals Networks to evaluate.
 * @param env Gymnasium environment.
 * @param seeds Seeds, one case each. Must not be empty.
 * @param resetOptions None, or a sequence with one env.reset(options=...) entry per seed.
 * @param settings Traversal limits and fallback/hold actions, see EpisodeSettings.
 * @param objectivesHook None, or callable(EpisodeResult) -> sequence of floats.
 * @param trackLineage If true, every case updates the lineage statistics
 *        (Network::updateLineageStats()).
 * @param successThreshold A case counts as a success for Network::lineageSuccessRate if its
 *        score is above this value (only with trackLineage).
 */
inline void evaluateEpisodes(std::vector<Network>& individuals, const py::object& env,
                             const std::vector<int>& seeds, const py::object& resetOptions,
                             const EpisodeSettings& settings, const py::object& objectivesHook,
                             bool trackLineage = false,
                             float successThreshold = std::numeric_limits<float>::infinity()) {
    if (seeds.empty()) {
        throw std::invalid_argument("gymnasium(): seeds must not be empty");
    }
    std::vector<py::object> options(seeds.size(), py::none());
    if (!resetOptions.is_none()) {
        py::sequence seq = resetOptions.cast<py::sequence>();
        if (py::len(seq) != seeds.size()) {
            throw std::invalid_argument("gymnasium(): resetOptions needs one entry per seed");
        }
        for (size_t c = 0; c < seeds.size(); ++c) options[c] = seq[c];
    }
    const bool withHook = !objectivesHook.is_none();

    for (auto& network : individuals) {
        network.caseScores.clear();
        network.episodeLog.clear();
        network.visitedNodesPerCase.clear();
        network.objectivesPerCase.clear();

        for (size_t c = 0; c < seeds.size(); ++c) {
            // Snapshot so a diff isolates this case's nodes (see Network::visitedNodesPerCase).
            // The first case resets the counters (newRun), so its baseline is zero, not the
            // previous evaluation's counts.
            std::vector<unsigned int> traverseCounterBefore(network.innerNodes.size(), 0);
            if (c > 0) {
                for (size_t n = 0; n < network.innerNodes.size(); ++n) {
                    traverseCounterBefore[n] = network.innerNodes[n].traverseCounter;
                }
            }

            std::vector<float> objectives =
                runEpisode(network, env, seeds[c], options[c], c == 0, settings, objectivesHook);

            std::vector<int> visitedThisCase;
            for (size_t n = 0; n < network.innerNodes.size(); ++n) {
                if (network.innerNodes[n].traverseCounter != traverseCounterBefore[n]) {
                    visitedThisCase.push_back(static_cast<int>(n));
                }
            }
            network.visitedNodesPerCase.push_back(std::move(visitedThisCase));
            network.caseScores.push_back(network.fitness);
            if (withHook) network.objectivesPerCase.push_back(std::move(objectives));
            if (trackLineage) {
                network.updateLineageStats(network.fitness, network.fitness > successThreshold);
            }
        }
        // Shared with parallel_eval.py, which stitches seed blocks together
        network.aggregateCaseResults();
    }
}

#endif // GYMNASIUM_WRAPPER_HPP
