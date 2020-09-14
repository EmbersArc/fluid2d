#include <Eigen/Dense>
#include <SFML/Graphics.hpp>

#include <iomanip>
#include <sstream>

const int GRIDSIZE = 256;
const double WINDOW_SCALING_FACTOR = 3.;
const double GAUSS_SEIDEL_TOLERANCE = 1e-4;
const int GAUSS_SEIDEL_ITER = 20;

const int FPS = 60;
const double VISC = 0.;
const double DIFF = 0.00002;
const double SOURCE = 20000.;
const double FORCE = 150.;
const double DISSOLVE = 0.005;

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXi;

void set_bnd(int b, MatrixXd &x)
{
    if (b == 1)
    {
        x.row(0) = -x.row(1);
        x.row(x.rows() - 1) = -x.row(x.rows() - 2);
    }
    else
    {
        x.row(0) = x.row(1);
        x.row(x.rows() - 1) = x.row(x.rows() - 2);
    }

    if (b == 2)
    {
        x.col(0) = -x.col(1);
        x.col(x.cols() - 1) = -x.col(x.cols() - 2);
    }
    else
    {
        x.col(0) = x.col(1);
        x.col(x.cols() - 1) = x.col(x.cols() - 2);
    }

    x(0, 0) = 0.5 * (x(1, 0) + x(0, 1));
    x(0, x.cols() - 1) = 0.5 * (x(1, x.cols() - 1) + x(0, x.cols() - 2));
    x(x.rows() - 1, 0) = 0.5 * (x(x.rows() - 2, 0) + x(x.rows() - 1, 1));
    x(x.rows() - 1, x.cols() - 1) = 0.5 * (x(x.rows() - 2, x.cols() - 1) + x(x.rows() - 1, x.cols() - 2));
}

void lin_solve(int b, MatrixXd &x, const MatrixXd &x0, double a, double c)
{
    MatrixXd x_last = x;

    const int N = x.rows() - 2;

    for (int i = 0; i < GAUSS_SEIDEL_ITER; i++)
    {
        x.block(1, 1, N, N) = (x0.block(1, 1, N, N) +
                               a * (x.block(0, 1, N, N) +
                                    x.block(2, 1, N, N) +
                                    x.block(1, 0, N, N) +
                                    x.block(1, 2, N, N))) /
                              c;
        set_bnd(b, x);

        if ((x_last - x).lpNorm<Eigen::Infinity>() < GAUSS_SEIDEL_TOLERANCE)
        {
            break;
        }

        x_last = x;
    }
}

void add_source(MatrixXd &d, const MatrixXd &s, double dt)
{
    d += dt * s;
}

void diffuse(int b, MatrixXd &x, const MatrixXd &x0, double diff, double dt)
{
    const int N = x.rows() - 2;
    const double a = dt * diff * N * N;
    lin_solve(b, x, x0, a, 1. + 4. * a);
}

void advect(int b, MatrixXd &d, const MatrixXd &d0,
            const MatrixXd &u, const MatrixXd &v, double dt)
{
    const int N = d.rows() - 2;

    const double dt0 = dt * N;

    for (int i = 1; i <= N; i++)
    {
        for (int j = 1; j <= N; j++)
        {
            double x = i - dt0 * u(i, j);
            double y = j - dt0 * v(i, j);

            x = std::clamp(x, 0.5, N + 0.5);
            y = std::clamp(y, 0.5, N + 0.5);

            const int i0 = int(x);
            const int j0 = int(y);

            const int i1 = i0 + 1;
            const int j1 = j0 + 1;

            const double s1 = x - i0;
            const double t1 = y - j0;

            const double s0 = 1. - s1;
            const double t0 = 1. - t1;

            d(i, j) = s0 * (t0 * d0(i0, j0) + t1 * d0(i0, j1)) +
                      s1 * (t0 * d0(i1, j0) + t1 * d0(i1, j1));
        }
    }

    set_bnd(b, d);
}

void project(MatrixXd &u, MatrixXd &v, MatrixXd &p, MatrixXd &div)
{
    const int N = div.rows() - 2;
    const double h = 1. / N;

    div.block(1, 1, N, N) = -0.5 * h *
                            (u.block(2, 1, N, N) -
                             u.block(0, 1, N, N) +
                             v.block(1, 2, N, N) -
                             v.block(1, 0, N, N));

    set_bnd(0, div);

    p.block(1, 1, N, N).setZero();
    set_bnd(0, p);

    lin_solve(0, p, div, 1., 4.);

    u.block(1, 1, N, N) -= 0.5 / h * (p.block(2, 1, N, N) - p.block(0, 1, N, N));
    v.block(1, 1, N, N) -= 0.5 / h * (p.block(1, 2, N, N) - p.block(1, 0, N, N));

    set_bnd(1, u);
    set_bnd(2, v);
}

void dens_step(MatrixXd &x, MatrixXd &x0,
               const MatrixXd &u, const MatrixXd &v,
               double diff, double dt)
{
    x *= (1. - DISSOLVE);
    add_source(x, x0, dt);
    diffuse(0, x0, x, diff, dt);
    advect(0, x, x0, u, v, dt);
}

void vel_step(MatrixXd &u, MatrixXd &v,
              MatrixXd &u0, MatrixXd &v0,
              double visc, double dt)
{
    add_source(u, u0, dt);
    add_source(v, v0, dt);

    diffuse(1, u0, u, visc, dt);
    diffuse(2, v0, v, visc, dt);

    project(u0, v0, u, v);

    advect(1, u, u0, u0, v0, dt);
    advect(2, v, v0, u0, v0, dt);

    project(u, v, u0, v0);
}

sf::Vector2i screen_coord_to_grid(sf::Vector2i coords)
{
    coords.x /= WINDOW_SCALING_FACTOR;
    coords.y /= WINDOW_SCALING_FACTOR;

    coords.x += 1;
    coords.y += 1;

    coords.x = std::clamp(coords.x, 1, GRIDSIZE - 2);
    coords.y = std::clamp(coords.y, 1, GRIDSIZE - 2);

    return {coords.x, GRIDSIZE - coords.y};
}

int main()
{
    // Velocity
    MatrixXd u(GRIDSIZE, GRIDSIZE);
    u.setZero();
    MatrixXd v(GRIDSIZE, GRIDSIZE);
    v.setZero();
    MatrixXd u_prev(GRIDSIZE, GRIDSIZE);
    u_prev.setZero();
    MatrixXd v_prev(GRIDSIZE, GRIDSIZE);
    v_prev.setZero();
    // Density
    MatrixXd d(GRIDSIZE, GRIDSIZE);
    d.setZero();
    MatrixXd d_prev(GRIDSIZE, GRIDSIZE);
    d_prev.setZero();

    // Drawing
    const uint WINDOW_SIZE = (GRIDSIZE - 2) * WINDOW_SCALING_FACTOR;
    sf::VideoMode mode(WINDOW_SIZE, WINDOW_SIZE);
    sf::RenderWindow window(mode, "Fluid", sf::Style::Close);
    window.setFramerateLimit(FPS);

    using MatrixXu32 = Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXu8 = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;
    using color_stride_map_t = Eigen::Map<MatrixXu8, 0, Eigen::InnerStride<4>>;
    MatrixXu32 buffer(GRIDSIZE - 2, GRIDSIZE - 2);
    color_stride_map_t mapped_red((uint8_t *)buffer.data() + 0, buffer.rows(), buffer.cols());
    color_stride_map_t mapped_green((uint8_t *)buffer.data() + 1, buffer.rows(), buffer.cols());
    color_stride_map_t mapped_blue((uint8_t *)buffer.data() + 2, buffer.rows(), buffer.cols());
    color_stride_map_t mapped_alpha((uint8_t *)buffer.data() + 3, buffer.rows(), buffer.cols());
    mapped_red.setConstant(0);
    mapped_green.setConstant(255);
    mapped_blue.setConstant(255);
    mapped_alpha.setConstant(255);

    sf::Texture texture;
    texture.create(buffer.rows(), buffer.cols());
    texture.setSmooth(true);

    sf::Vector2i last_mouse_position(-1, -1);
    bool left_pressed = false;
    bool right_pressed = false;

    uint frames = 0;
    double current_framerate;
    sf::Clock clock;
    sf::Font font;
    if (not font.loadFromFile("../DejaVuSans-Bold.ttf"))
    {
        throw std::runtime_error("Font file not found!");
    }
    sf::Text text;
    text.setFont(font);
    const uint text_height = WINDOW_SIZE / 20;
    text.setCharacterSize(text_height);
    text.setFillColor(sf::Color(100, 100, 100));
    text.setPosition(text_height * 0.4, WINDOW_SIZE - text_height * 1.4);

    while (window.isOpen())
    {
        // Update
        {
            const double dt = 1. / FPS;
            vel_step(u, v, u_prev, v_prev, VISC, dt);
            dens_step(d, d_prev, u, v, DIFF, dt);

            d_prev.setZero();
            u_prev.setZero();
            v_prev.setZero();

            const int center = GRIDSIZE / 2;
            const int lower = GRIDSIZE / 5;
            const double source = 10000.;
            const double force = 50.;

            d_prev.block<2, 2>(lower, center).setConstant(source);
            u_prev.block<2, 2>(lower, center).setConstant(force);
            d_prev.block<2, 2>(d_prev.rows() - lower, center).setConstant(source);
            u_prev.block<2, 2>(u_prev.rows() - lower, center).setConstant(-force);
        }

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::MouseButtonPressed)
            {
                if (event.mouseButton.button == sf::Mouse::Right)
                {
                    right_pressed = true;
                }
                if (event.mouseButton.button == sf::Mouse::Left)
                {
                    left_pressed = true;
                }
            }

            if (event.type == sf::Event::MouseButtonReleased)
            {
                if (event.mouseButton.button == sf::Mouse::Right)
                {
                    right_pressed = false;
                }
                if (event.mouseButton.button == sf::Mouse::Left)
                {
                    left_pressed = false;
                }
            }
        }

        if (right_pressed)
        {
            const sf::Vector2i mouse_position = sf::Mouse::getPosition(window);
            const sf::Vector2i idx = screen_coord_to_grid(mouse_position);

            d_prev(idx.x, idx.y) = SOURCE;
        }
        if (left_pressed)
        {
            const sf::Vector2i mouse_position = sf::Mouse::getPosition(window);
            const sf::Vector2i idx = screen_coord_to_grid(mouse_position);

            if (last_mouse_position != sf::Vector2i(-1, -1))
            {
                const double dx = last_mouse_position.x - mouse_position.x;
                const double dy = last_mouse_position.y - mouse_position.y;
                u_prev(idx.x, idx.y) = -FORCE * dx / WINDOW_SCALING_FACTOR;
                v_prev(idx.x, idx.y) = FORCE * dy / WINDOW_SCALING_FACTOR;
            }
            last_mouse_position = mouse_position;
        }
        else
        {
            last_mouse_position = {-1, -1};
        }

        window.clear(sf::Color::Black);

        mapped_alpha << d.block(1, 1, GRIDSIZE - 2, GRIDSIZE - 2).cwiseMin(255.).cast<uint8_t>();

        texture.update((uint8_t *)buffer.data());
        sf::Sprite sprite(texture);
        sprite.setOrigin(sprite.getLocalBounds().width / 2.,
                         sprite.getLocalBounds().height / 2.);
        sprite.setPosition(float(window.getSize().x) / 2, float(window.getSize().y) / 2);
        sprite.setScale(-WINDOW_SCALING_FACTOR, WINDOW_SCALING_FACTOR);
        sprite.setRotation(-180.);
        window.draw(sprite);

        sf::Time elapsed = clock.getElapsedTime();
        if (elapsed.asSeconds() > 0.25)
        {
            current_framerate = frames / elapsed.asSeconds();
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << current_framerate;
            text.setString(stream.str());

            std::string s = stream.str();
            frames = 0;
            clock.restart();
        }

        window.draw(text);

        window.display();
        frames += 1;
    }

    return 0;
}