#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <memory> // For smart pointers

// Include Box2D headers
#include <Box2D/Box2D.h>

// Include a graphics library (e.g., SFML)
// #include <SFML/Graphics.hpp> // Example for SFML

// --- Constants (equivalent to Python's global constants) ---
const float FPS = 50.0f;
const float SCALE = 30.0f;

const float MAIN_ENGINE_POWER = 13.0f;
const float SIDE_ENGINE_POWER = 0.6f;

const float INITIAL_RANDOM = 1000.0f;

// Lander polygon vertices (scaled later)
// The 'f' suffix is crucial for float literals.
const std::vector<b2Vec2> LANDER_POLY_RAW = {
    {-14.0f, +17.0f}, {-17.0f, 0.0f}, {-17.0f, -10.0f},
    {+17.0f, -10.0f}, {+17.0f, 0.0f}, {+14.0f, +17.0f}
};

const float LEG_AWAY = 20.0f;
const float LEG_DOWN = 18.0f;
const float LEG_W = 2.0f;
const float LEG_H = 8.0f;
const float LEG_SPRING_TORQUE = 40.0f;

const float SIDE_ENGINE_HEIGHT = 14.0f;
const float SIDE_ENGINE_AWAY = 12.0f;
const float MAIN_ENGINE_Y_LOCATION = 4.0f;

const int VIEWPORT_W = 600;
const int VIEWPORT_H = 400;

// --- ContactDetector Class (inherits from b2ContactListener) ---
class ContactDetector : public b2ContactListener {
public:
    // Forward declaration to avoid circular dependency if LunarLander needs to be defined first
    // For simplicity, we'll assume LunarLander pointer is passed.
    class LunarLander; // Forward declaration

    LunarLander* env; // Pointer to the environment

    ContactDetector(LunarLander* lunarLanderEnv) : env(lunarLanderEnv) {}

    void BeginContact(b2Contact* contact) override;
    void EndContact(b2Contact* contact) override;
};

// --- LunarLander Class ---
class LunarLander {
public:
    // Box2D World and Bodies
    std::unique_ptr<b2World> world; // Using unique_ptr for automatic memory management
    b2Body* moon;
    b2Body* lander;
    std::vector<b2Body*> legs;
    std::vector<b2Body*> particles; // Particles for engine exhaust

    // Game state variables
    bool game_over;
    float prev_shaping;

    // Environment parameters
    float gravity_val;
    bool enable_wind;
    float wind_power_val;
    float turbulence_power_val;
    
    // Random number generation
    std::mt19937 rng; // Mersenne Twister engine
    std::uniform_real_distribution<float> uniform_dist;
    std::uniform_int_distribution<int> int_dist; // For wind indices

    int wind_idx;
    int torque_idx;

    // Terrain
    float helipad_x1, helipad_x2, helipad_y;
    std::vector<std::vector<b2Vec2>> sky_polys; // For rendering the sky

    // Rendering (SFML example)
    // sf::RenderWindow* screen;
    // sf::Clock clock;
    // sf::Texture surf_texture; // Using a texture for drawing to
    // sf::Sprite surf_sprite;

    std::string render_mode; // "human", "rgb_array", or "none"

    bool continuous_actions; // Discrete or continuous action space

    // Constructor
    LunarLander(std::string renderMode = "none", bool continuous = false, float gravity = -10.0f,
                bool enableWind = false, float windPower = 15.0f, float turbulencePower = 1.5f)
        : gravity_val(gravity), enable_wind(enableWind), wind_power_val(windPower),
          turbulence_power_val(turbulencePower), game_over(false), prev_shaping(0.0f),
          render_mode(renderMode), continuous_actions(continuous), rng(std::random_device{}()) {
        
        if (gravity_val <= -12.0f || gravity_val >= 0.0f) {
            std::cerr << "Error: gravity (current value: " << gravity_val << ") must be between -12 and 0" << std::endl;
            exit(1); // Or throw an exception
        }
        if (wind_power_val < 0.0f || wind_power_val > 20.0f) {
            std::cout << "Warning: wind_power value is recommended to be between 0.0 and 20.0, (current value: " << wind_power_val << ")" << std::endl;
        }
        if (turbulence_power_val < 0.0f || turbulence_power_val > 2.0f) {
            std::cout << "Warning: turbulence_power value is recommended to be between 0.0 and 2.0, (current value: " << turbulence_power_val << ")" << std::endl;
        }

        // Initialize distributions for RNG
        uniform_dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
        int_dist = std::uniform_int_distribution<int>(-9999, 9999);
    }

    // Destructor to clean up Box2D bodies and other resources
    ~LunarLander() {
        _destroy();
    }

    // Equivalent of _destroy() in Python
    void _destroy() {
        if (!world) return; // If world not initialized, nothing to destroy

        // No need to set contactListener to nullptr if world is destroyed
        // Box2D automatically cleans up bodies when world is destroyed if created through it.
        // However, if you add custom contact listeners, ensure they are managed.
        
        // Clear particles explicitly if they are managed outside of the main bodies
        for (b2Body* p : particles) {
            if (p) world->DestroyBody(p);
        }
        particles.clear();
        
        // Bodies are destroyed when the world is destroyed
        moon = nullptr;
        lander = nullptr;
        legs.clear(); // Pointers become invalid after world destruction
    }

    // Equivalent of reset() in Python
    // Returns observation vector and info map
    std::pair<std::vector<float>, std::map<std::string, float>> reset(int seed = -1, std::map<std::string, float> options = {}) {
        _destroy(); // Clean up previous world

        // Create a new Box2D world for each reset (as in Python workaround)
        world = std::make_unique<b2World>(b2Vec2(0.0f, gravity_val));
        
        // Set up contact listener
        // The ContactDetector needs a pointer to this LunarLander instance.
        // Be careful with object lifetimes here. A shared_ptr might be safer for env.
        world->SetContactListener(new ContactDetector(this)); // Box2D takes ownership of listener

        game_over = false;
        prev_shaping = 0.0f;

        // Reset RNG for reproducibility if seed is provided
        if (seed != -1) {
            rng.seed(seed);
        }

        // Create Terrain
        const int CHUNKS = 11;
        std::vector<float> height(CHUNKS + 1);
        for (int i = 0; i <= CHUNKS; ++i) {
            height[i] = uniform_dist(rng) * (VIEWPORT_H / SCALE / 4); // np.uniform(0, H/2) equivalent
        }

        std::vector<float> chunk_x(CHUNKS);
        for (int i = 0; i < CHUNKS; ++i) {
            chunk_x[i] = (VIEWPORT_W / SCALE / (CHUNKS - 1)) * i;
        }

        helipad_x1 = chunk_x[CHUNKS / 2 - 1];
        helipad_x2 = chunk_x[CHUNKS / 2 + 1];
        helipad_y = (VIEWPORT_H / SCALE) / 4;

        height[CHUNKS / 2 - 2] = helipad_y;
        height[CHUNKS / 2 - 1] = helipad_y;
        height[CHUNKS / 2 + 0] = helipad_y;
        height[CHUNKS / 2 + 1] = helipad_y;
        height[CHUNKS / 2 + 2] = helipad_y;

        std::vector<float> smooth_y(CHUNKS);
        for (int i = 1; i < CHUNKS; ++i) {
            smooth_y[i] = 0.33f * (height[i - 1] + height[i + 0] + height[i + 1]);
        }
        smooth_y[0] = 0.33f * (height[0] + height[1] + height[2]); // Handle boundary for i=0
        smooth_y[CHUNKS-1] = 0.33f * (height[CHUNKS-2] + height[CHUNKS-1] + height[CHUNKS]); // Handle boundary for last chunk

        b2BodyDef moonDef;
        moon = world->CreateBody(&moonDef); // Static body
        
        sky_polys.clear();
        for (int i = 0; i < CHUNKS - 1; ++i) {
            b2Vec2 p1(chunk_x[i], smooth_y[i]);
            b2Vec2 p2(chunk_x[i + 1], smooth_y[i + 1]);
            moon->CreateFixture(new b2EdgeShape(), 0.0f); // Fix: Edge shape needs to be configured
            b2EdgeShape edge;
            edge.SetTwoSided(p1, p2);
            b2FixtureDef fd;
            fd.shape = &edge;
            fd.density = 0.0f;
            fd.friction = 0.1f;
            moon->CreateFixture(&fd);

            sky_polys.push_back({p1, p2, b2Vec2(p2.x, VIEWPORT_H / SCALE), b2Vec2(p1.x, VIEWPORT_H / SCALE)});
        }
        // moon->color1 and color2 are for rendering, handle in render()

        // Create Lander body
        float initial_y = VIEWPORT_H / SCALE;
        float initial_x = VIEWPORT_W / SCALE / 2;

        b2BodyDef landerDef;
        landerDef.type = b2_dynamicBody;
        landerDef.position.Set(initial_x, initial_y);
        landerDef.angle = 0.0f;
        lander = world->CreateBody(&landerDef);

        b2PolygonShape landerShape;
        std::vector<b2Vec2> lander_poly_scaled;
        for (const auto& p : LANDER_POLY_RAW) {
            lander_poly_scaled.push_back(b2Vec2(p.x / SCALE, p.y / SCALE));
        }
        landerShape.Set(lander_poly_scaled.data(), lander_poly_scaled.size());

        b2FixtureDef landerFd;
        landerFd.shape = &landerShape;
        landerFd.density = 5.0f;
        landerFd.friction = 0.1f;
        landerFd.categoryBits = 0x0010;
        landerFd.maskBits = 0x001; // collide only with ground
        landerFd.restitution = 0.0f;
        lander->CreateFixture(&landerFd);

        // Apply initial random impulse
        float impulse_x = uniform_dist(rng) * INITIAL_RANDOM;
        float impulse_y = uniform_dist(rng) * INITIAL_RANDOM;
        lander->ApplyForceToCenter(b2Vec2(impulse_x, impulse_y), true);

        // Wind initialization
        if (enable_wind) {
            wind_idx = int_dist(rng);
            torque_idx = int_dist(rng);
        }

        // Create Lander Legs
        legs.clear();
        for (int i : {-1, 1}) {
            b2BodyDef legDef;
            legDef.type = b2_dynamicBody;
            legDef.position.Set(initial_x - i * LEG_AWAY / SCALE, initial_y);
            legDef.angle = (i * 0.05f);
            b2Body* leg = world->CreateBody(&legDef);

            b2PolygonShape legShape;
            legShape.SetAsBox(LEG_W / SCALE, LEG_H / SCALE);

            b2FixtureDef legFd;
            legFd.shape = &legShape;
            legFd.density = 1.0f;
            legFd.restitution = 0.0f;
            legFd.categoryBits = 0x0020;
            legFd.maskBits = 0x001;
            leg->CreateFixture(&legFd);

            // Store ground contact status (requires custom property or a map)
            // For now, let's just make it a public member of leg, similar to Python's approach
            // A more robust C++ solution would be a `std::map<b2Body*, bool> ground_contact_status;`
            // and associate this with the body in the contact listener.
            // For simplicity, let's add a `bool ground_contact` member to the `b2Body` user data.
            struct LegUserData {
                bool ground_contact = false;
            };
            leg->SetUserData(new LegUserData()); // Remember to delete this in _destroy or destructor

            b2RevoluteJointDef rjd;
            rjd.bodyA = lander;
            rjd.bodyB = leg;
            rjd.localAnchorA.Set(0, 0); // Anchor on lander at its center of mass
            rjd.localAnchorB.Set(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE); // Anchor on leg
            rjd.enableMotor = true;
            rjd.enableLimit = true;
            rjd.maxMotorTorque = LEG_SPRING_TORQUE;
            rjd.motorSpeed = +0.3f * i;

            if (i == -1) {
                rjd.lowerAngle = (+0.9f - 0.5f);
                rjd.upperAngle = +0.9f;
            } else {
                rjd.lowerAngle = -0.9f;
                rjd.upperAngle = -0.9f + 0.5f;
            }
            world->CreateJoint(&rjd);
            legs.push_back(leg);
        }
        
        // Handle rendering initialization
        // if (render_mode == "human" && screen == nullptr) {
        //     screen = new sf::RenderWindow(sf::VideoMode(VIEWPORT_W, VIEWPORT_H), "Lunar Lander");
        //     surf_texture.create(VIEWPORT_W, VIEWPORT_H);
        //     surf_sprite.setTexture(surf_texture);
        // }


        // Return initial observation and empty info map
        // The Python step function calls itself once here with action 0 to get the initial state
        // We'll mimic that by calling step(0) and returning its result.
        return step(0); 
    }

    // Equivalent of _create_particle in Python
    b2Body* _create_particle(float mass, float x, float y, float ttl) {
        b2BodyDef bd;
        bd.type = b2_dynamicBody;
        bd.position.Set(x, y);
        bd.angle = 0.0f;
        b2Body* p = world->CreateBody(&bd);

        b2CircleShape cs;
        cs.m_radius = 2.0f / SCALE;
        cs.m_p.Set(0, 0);

        b2FixtureDef fd;
        fd.shape = &cs;
        fd.density = mass;
        fd.friction = 0.1f;
        fd.categoryBits = 0x0100;
        fd.maskBits = 0x001; // collide only with ground
        fd.restitution = 0.3f;
        p->CreateFixture(&fd);

        // Store TTL (similar to Python's dynamic attribute)
        // A struct for particle data would be better
        struct ParticleUserData {
            float ttl;
            // Add color info if needed for rendering
            unsigned char color1_r, color1_g, color1_b;
            unsigned char color2_r, color2_g, color2_b;
        };
        ParticleUserData* userData = new ParticleUserData();
        userData->ttl = ttl;
        p->SetUserData(userData);
        particles.push_back(p);
        _clean_particles(false);
        return p;
    }

    // Equivalent of _clean_particles in Python
    void _clean_particles(bool all_particle) {
        // Iterate backwards to safely remove elements while iterating
        for (auto it = particles.begin(); it != particles.end(); ) {
            // Retrieve user data
            ParticleUserData* userData = static_cast<ParticleUserData*>((*it)->GetUserData());
            if (userData && (all_particle || userData->ttl < 0)) {
                world->DestroyBody(*it);
                delete userData; // Delete the user data
                it = particles.erase(it); // Remove from vector
            } else {
                ++it;
            }
        }
    }

    // Equivalent of step() in Python
    // Returns observation, reward, terminated, truncated, info
    std::tuple<std::vector<float>, float, bool, bool, std::map<std::string, float>> step(int action_discrete) {
        // Use overloads for continuous vs. discrete actions
        // For brevity, only implementing discrete for now.
        // For continuous, you'd need `std::tuple<std::vector<float>, float, bool, bool, std::map<std::string, float>> step(std::vector<float> action_continuous)`.

        if (!lander) {
            std::cerr << "Error: You forgot to call reset()" << std::endl;
            exit(1);
        }

        // Apply wind effects
        if (enable_wind && !((static_cast<LegUserData*>(legs[0]->GetUserData()))->ground_contact || (static_cast<LegUserData*>(legs[1]->GetUserData()))->ground_contact)) {
            float wind_mag = std::tanh(std::sin(0.02f * wind_idx) + (std::sin(M_PI * 0.01f * wind_idx))) * wind_power_val;
            wind_idx += 1;
            lander->ApplyForceToCenter(b2Vec2(wind_mag, 0.0f), true);

            float torque_mag = std::tanh(std::sin(0.02f * torque_idx) + (std::sin(M_PI * 0.01f * torque_idx))) * turbulence_power_val;
            torque_idx += 1;
            lander->ApplyTorque(torque_mag, true);
        }

        float m_power = 0.0f;
        float s_power = 0.0f;

        // Engine Impulses
        b2Vec2 tip(std::sin(lander->GetAngle()), std::cos(lander->GetAngle()));
        b2Vec2 side(-tip.y, tip.x);

        float dispersion[2] = {uniform_dist(rng) / SCALE, uniform_dist(rng) / SCALE};

        if ((continuous_actions && action_discrete > 0) || (!continuous_actions && action_discrete == 2)) {
            // Main engine (assuming discrete action = 2 for main engine)
            if (continuous_actions) {
                // Not implementing continuous action logic here, but it would clip and scale action[0]
                // m_power = (std::fmax(0.0f, action[0]) + 1.0f) * 0.5f; 
            } else {
                m_power = 1.0f;
            }

            float ox = tip.x * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) + side.x * dispersion[1];
            float oy = -tip.y * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) - side.y * dispersion[1];

            b2Vec2 impulse_pos(lander->GetPosition().x + ox, lander->GetPosition().y + oy);

            if (render_mode != "none") {
                b2Body* p = _create_particle(3.5f, impulse_pos.x, impulse_pos.y, m_power);
                p->ApplyLinearImpulse(b2Vec2(ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power), impulse_pos, true);
            }
            lander->ApplyLinearImpulse(b2Vec2(-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power), impulse_pos, true);
        }

        if ((continuous_actions && std::abs(action_discrete) > 0.5f) || (!continuous_actions && (action_discrete == 1 || action_discrete == 3))) {
            // Side engines (assuming discrete action = 1 for left, 3 for right)
            int direction;
            if (continuous_actions) {
                // Not implementing continuous action logic here, but it would set direction and s_power
                // direction = (action_continuous[1] > 0) ? 1 : -1;
                // s_power = std::fmax(0.5f, std::abs(action_continuous[1]));
            } else {
                direction = (action_discrete == 1) ? -1 : 1;
                s_power = 1.0f;
            }

            float ox = tip.x * dispersion[0] + side.x * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE);
            float oy = -tip.y * dispersion[0] - side.y * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE);

            b2Vec2 impulse_pos(lander->GetPosition().x + ox - tip.x * 17.0f / SCALE, lander->GetPosition().y + oy + tip.y * SIDE_ENGINE_HEIGHT / SCALE);

            if (render_mode != "none") {
                b2Body* p = _create_particle(0.7f, impulse_pos.x, impulse_pos.y, s_power);
                p->ApplyLinearImpulse(b2Vec2(ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power), impulse_pos, true);
            }
            lander->ApplyLinearImpulse(b2Vec2(-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power), impulse_pos, true);
        }

        // Simulate the world
        world->Step(1.0f / FPS, 6, 2);

        // Get state
        b2Vec2 pos = lander->GetPosition();
        b2Vec2 vel = lander->GetLinearVelocity();

        std::vector<float> state = {
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            lander->GetAngle(),
            20.0f * lander->GetAngularVelocity() / FPS,
            (static_cast<LegUserData*>(legs[0]->GetUserData()))->ground_contact ? 1.0f : 0.0f,
            (static_cast<LegUserData*>(legs[1]->GetUserData()))->ground_contact ? 1.0f : 0.0f
        };

        float reward = 0.0f;
        float shaping = 
            -100.0f * std::sqrt(state[0] * state[0] + state[1] * state[1])
            - 100.0f * std::sqrt(state[2] * state[2] + state[3] * state[3])
            - 100.0f * std::abs(state[4])
            + 10.0f * state[6]
            + 10.0f * state[7];

        if (prev_shaping != 0.0f) { // Check if not the first step
            reward = shaping - prev_shaping;
        }
        prev_shaping = shaping;

        reward -= m_power * 0.30f;
        reward -= s_power * 0.03f;

        bool terminated = false;
        bool truncated = false; // Gymnasium's concept of truncation

        if (game_over || std::abs(state[0]) >= 1.0f) {
            terminated = true;
            reward = -100.0f;
        }
        if (!lander->IsAwake()) {
            terminated = true;
            reward = +100.0f;
        }

        if (render_mode == "human") {
            render();
        }

        std::map<std::string, float> info; // Empty info map for now
        return std::make_tuple(state, reward, terminated, truncated, info);
    }

    // Equivalent of render() in Python
    void render() {
        // This is where SFML/SDL/etc. rendering code would go.
        // It would involve:
        // 1. Clearing the window.
        // 2. Drawing the sky polygons.
        // 3. Iterating through lander, legs, and particles to draw their shapes
        //    (polygons for lander/legs, circles for particles) using their positions, angles, and colors.
        // 4. Drawing the helipad flags.
        // 5. Flipping the surface (for human mode).
        // 6. Displaying the rendered image.
        // 7. Handling events (e.g., closing the window).
        if (render_mode == "none") {
            return;
        }
        // Example (SFML conceptual, not fully implemented)
        /*
        if (!screen) {
            std::cerr << "Render window not initialized." << std::endl;
            return;
        }

        screen->clear(sf::Color::White); // Equivalent to pygame.draw.rect(self.surf, (255,255,255), self.surf.get_rect())

        // Update particle colors and TTL
        for (b2Body* obj : particles) {
            ParticleUserData* userData = static_cast<ParticleUserData*>(obj->GetUserData());
            if (userData) {
                userData->ttl -= 0.15f;
                userData->color1_r = static_cast<unsigned char>(std::max(0.2f, 0.15f + userData->ttl) * 255);
                userData->color1_g = static_cast<unsigned char>(std::max(0.2f, 0.5f * userData->ttl) * 255);
                userData->color1_b = static_cast<unsigned char>(std::max(0.2f, 0.5f * userData->ttl) * 255);
                // Similar for color2
            }
        }
        _clean_particles(false);

        // Draw sky polygons
        for (const auto& p : sky_polys) {
            // Convert b2Vec2 to sf::Vector2f and draw polygon
            sf::ConvexShape convex;
            convex.setPointCount(p.size());
            for (size_t i = 0; i < p.size(); ++i) {
                convex.setPoint(i, sf::Vector2f(p[i].x * SCALE, VIEWPORT_H - p[i].y * SCALE)); // Flip Y-axis for SFML
            }
            convex.setFillColor(sf::Color::Black);
            screen->draw(convex);
        }

        // Draw bodies (lander, legs, particles)
        // This would be a loop over drawlist (lander + legs) and particles
        // and drawing their shapes using Box2D's fixture information.
        // For polygon shapes: convert vertices, scale, and draw.
        // For circle shapes: convert position, scale radius, and draw.

        // Draw helipad flags
        for (float x_coord : {helipad_x1, helipad_x2}) {
            float x_scaled = x_coord * SCALE;
            float flagy1_scaled = helipad_y * SCALE;
            float flagy2_scaled = flagy1_scaled + 50.0f; // 50px length

            sf::RectangleShape line(sf::Vector2f(1, 50));
            line.setPosition(x_scaled, VIEWPORT_H - flagy2_scaled); // Flip Y-axis
            line.setFillColor(sf::Color::White);
            screen->draw(line);

            sf::ConvexShape flag_triangle;
            flag_triangle.setPointCount(3);
            flag_triangle.setPoint(0, sf::Vector2f(x_scaled, VIEWPORT_H - flagy2_scaled));
            flag_triangle.setPoint(1, sf::Vector2f(x_scaled, VIEWPORT_H - (flagy2_scaled - 10)));
            flag_triangle.setPoint(2, sf::Vector2f(x_scaled + 25, VIEWPORT_H - (flagy2_scaled - 5)));
            flag_triangle.setFillColor(sf::Color(204, 204, 0));
            screen->draw(flag_triangle);
        }

        if (render_mode == "human") {
            screen->display();
        }
        */
    }

    // Equivalent of close() in Python
    void close() {
        // Clean up rendering resources
        // if (screen) {
        //     screen->close();
        //     delete screen;
        //     screen = nullptr;
        // }
    }
};

// --- Implement ContactDetector methods outside of class for clarity ---
// (Requires LunarLander to be fully defined or forward declared with members)
void ContactDetector::BeginContact(b2Contact* contact) {
    // Check if lander is involved in contact
    if (env->lander == contact->GetFixtureA()->GetBody() || env->lander == contact->GetFixtureB()->GetBody()) {
        env->game_over = true;
    }

    // Check if any leg is involved in contact
    for (b2Body* leg : env->legs) {
        if (leg == contact->GetFixtureA()->GetBody() || leg == contact->GetFixtureB()->GetBody()) {
            // Cast user data back to LegUserData
            LunarLander::LegUserData* userData = static_cast<LunarLander::LegUserData*>(leg->GetUserData());
            if (userData) {
                userData->ground_contact = true;
            }
        }
    }
}

void ContactDetector::EndContact(b2Contact* contact) {
    for (b2Body* leg : env->legs) {
        if (leg == contact->GetFixtureA()->GetBody() || leg == contact->GetFixtureB()->GetBody()) {
            LunarLander::LegUserData* userData = static_cast<LunarLander::LegUserData*>(leg->GetUserData());
            if (userData) {
                userData->ground_contact = false;
            }
        }
    }
}


// --- Heuristic function (outside the class as in Python) ---
// This would take the state vector and return an action.
// For discrete, it would return an int (0-3).
// For continuous, it would return a std::vector<float> of size 2.
int heuristic_discrete(const std::vector<float>& s) {
    float angle_targ = s[0] * 0.5f + s[2] * 1.0f;
    if (angle_targ > 0.4f) angle_targ = 0.4f;
    if (angle_targ < -0.4f) angle_targ = -0.4f;
    // float hover_targ = 0.55f * std::abs(s[0]); // Not used for discrete action directly

    int action = 0; // Nop

    if (s[6] || s[7]) { // If legs are touching ground
        if (s[1] > 0) { // If still above helipad
            action = 2; // Fire main engine to go down slower
        } else {
            action = 0; // Landed, do nothing
        }
    } else { // In air
        if (angle_targ < 0 && s[4] < angle_targ) {
            action = 3; // Fire right engine to turn left
        } else if (angle_targ > 0 && s[4] > angle_targ) {
            action = 1; // Fire left engine to turn right
        } else { // Angle is good, control vertical speed
            if (s[3] < -0.5f) { // Too fast downwards
                action = 2; // Fire main engine
            } else {
                action = 0; // Nop or slight corrections
            }
        }
    }
    return action;
}

// --- Main function for demonstration (like Python's `if __name__ == "__main__":`) ---
int main() {
    // Example usage:
    LunarLander env("human", false); // "human" for rendering, false for discrete actions

    std::pair<std::vector<float>, std::map<std::string, float>> reset_result = env.reset();
    std::vector<float> state = reset_result.first;

    float total_reward = 0.0f;
    bool terminated = false;
    bool truncated = false;
    int steps = 0;

    std::cout << "Starting Lunar Lander simulation (C++ version)..." << std::endl;

    while (!terminated && !truncated) {
        int action = heuristic_discrete(state); // Get action from heuristic
        
        std::tuple<std::vector<float>, float, bool, bool, std::map<std::string, float>> step_result = env.step(action);
        
        state = std::get<0>(step_result);
        float reward = std::get<1>(step_result);
        terminated = std::get<2>(step_result);
        truncated = std::get<3>(step_result);
        // info = std::get<4>(step_result); // Not used in this simple example

        total_reward += reward;
        steps++;

        // You'd want some delay here if not using SFML's clock or similar for "human" render mode
        // For example, using std::this_thread::sleep_for(std::chrono::milliseconds(1000 / (int)FPS));

        // For console output, you can print state or reward every few steps
        if (steps % 50 == 0) {
            std::cout << "Step: " << steps << ", Total Reward: " << total_reward << ", X: " << state[0] << ", Y: " << state[1] << std::endl;
        }
    }

    std::cout << "Simulation finished in " << steps << " steps." << std::endl;
    std::cout << "Final Reward: " << total_reward << std::endl;
    std::cout << "Terminated: " << (terminated ? "True" : "False") << ", Truncated: " << (truncated ? "True" : "False") << std::endl;

    env.close(); // Clean up rendering resources

    return 0;
}
