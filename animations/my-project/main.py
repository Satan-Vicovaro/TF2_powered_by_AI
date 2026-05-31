from manim import *
import numpy as np


class Scene1_ShooterAndTarget(ThreeDScene):
    def construct(self):
        # 1. Inicjalizacja osi 3D
        axes = ThreeDAxes(
            x_range=[-1000, 1000, 500],
            y_range=[-1000, 1000, 500],
            z_range=[0, 1200, 300],
            x_length=10,
            y_length=10,
            z_length=3.2,
            axis_config={"stroke_width": 2},
        )

        # 2. Definicja pozycji: odpowiadające realiom mapy TF2
        shooter_coords = (-600, -600, 100)
        target_coords = (600, 600, 500)

        shooter_pos = axes.c2p(*shooter_coords)
        target_pos = axes.c2p(*target_coords)

        # POPRAWKA KOLORÓW: Shooter = RED, Target = BLUE
        shooter_dot = Dot3D(point=shooter_pos, color=RED, radius=0.08)
        target_dot = Dot3D(point=target_pos, color=BLUE, radius=0.08)

        # Trackery przezroczystości do kontrolowania pojawiania się etykiet
        shooter_opacity = ValueTracker(0.0)
        target_opacity = ValueTracker(0.0)

        # Dynamiczne etykiety automatycznie kontrujące ruch kamery (brak efektu lustra i poprawne skalowanie z zoomem)
        shooter_label = always_redraw(
            lambda: Text(f"Shooter\n{shooter_coords}", color=RED)
            .scale(0.2)
            .set_opacity(shooter_opacity.get_value())
            .move_to(shooter_pos + np.array([0, 0, 0.35]))
            .rotate(self.camera.get_phi(), RIGHT)
            .rotate(self.camera.get_theta() + 90 * DEGREES, OUT)
        )

        target_label = always_redraw(
            lambda: Text(f"Target\n{target_coords}", color=BLUE)
            .scale(0.2)
            .set_opacity(target_opacity.get_value())
            .move_to(target_pos + np.array([0, 0, 0.35]))
            .rotate(self.camera.get_phi(), RIGHT)
            .rotate(self.camera.get_theta() + 90 * DEGREES, OUT)
        )

        # 3. Widok początkowy (obniżona kamera, bardziej w lewo, mniejszy zoom)
        self.set_camera_orientation(phi=75 * DEGREES, theta=15 * DEGREES, zoom=0.45)
        self.play(Create(axes))
        self.wait(0.5)

        # 4. Dodanie strzelca
        self.add(shooter_label)
        self.play(FadeIn(shooter_dot))
        self.wait(0.5)

        # Zbliżenie na strzelca i płynne pokazanie etykiety
        self.move_camera(frame_center=shooter_pos + np.array([0, 0, 0.1]), zoom=1.2, run_time=1.5)
        self.play(shooter_opacity.animate.set_value(1.0), run_time=0.5)
        self.wait(0.2)

        # Powrót do widoku ogólnego
        self.move_camera(frame_center=ORIGIN, zoom=0.45, run_time=1.5)
        self.wait(0.5)

        # 5. Dodanie celu
        self.add(target_label)
        self.play(FadeIn(target_dot))
        self.wait(0.5)

        # Zbliżenie na cel i płynne pokazanie etykiety
        self.move_camera(frame_center=target_pos + np.array([0, 0, 0.1]), zoom=1.2, run_time=1.5)
        self.play(target_opacity.animate.set_value(1.0), run_time=0.5)
        self.wait(0.2)

        # 6. Odjazd kamery do widoku ogólnego i obrót (etykiety automatycznie podążają za obrotem)
        midpoint = (shooter_pos + target_pos) / 2
        self.move_camera(frame_center=midpoint, zoom=0.35, theta=105 * DEGREES, run_time=3)
        self.wait(2)


class Scene2_NeuralNetwork(Scene):
    def construct(self):
        # 1. Tytuł
        title = Text("Agent DDPG - Sieć Neuronowa").to_edge(UP).scale(0.8)
        self.play(Write(title))
        self.wait(0.5)

        # 3. Rysowanie warstw sieci neuronowej - uproszczone
        layers = [6, 10, 10, 2]
        nodes = VGroup()
        edges = VGroup()

        layer_spacing = 0.7
        node_spacing = 0.08
        node_radius = 0.05

        # Budowanie węzłów - wyśrodkowane, przesunięte w górę by zrobić miejsce na etykiety
        net_offset = LEFT * (len(layers) - 1) * layer_spacing / 2 + UP * 0.3
        for i, num_nodes in enumerate(layers):
            layer_nodes = VGroup()
            for j in range(num_nodes):
                node = Circle(radius=node_radius, color=BLUE, fill_opacity=0.3)
                layer_nodes.add(node)
            layer_nodes.arrange(DOWN, buff=node_spacing)
            layer_nodes.move_to(net_offset + RIGHT * (i * layer_spacing))
            nodes.add(layer_nodes)

        # Budowanie połączeń - bez wypełnienia (tylko linie)
        for i in range(len(layers) - 1):
            layer_edges = VGroup()
            for node1 in nodes[i]:
                for node2 in nodes[i + 1]:
                    edge = Line(
                        node1.get_right(),
                        node2.get_left(),
                        color=WHITE,
                        stroke_opacity=0.05,
                        stroke_width=0.5,
                    )
                    layer_edges.add(edge)
            edges.add(layer_edges)

        self.play(Create(nodes), run_time=1)
        self.play(Create(edges), run_time=0.8)
        self.wait(0.3)

        # Etykiety warstw
        layer_labels = VGroup(
            Text("Input\n(6)", font_size=10).next_to(nodes[0], DOWN, buff=0.2),
            Text("Hidden\n(256)", font_size=10).next_to(nodes[1], DOWN, buff=0.2),
            Text("Hidden\n(256)", font_size=10).next_to(nodes[2], DOWN, buff=0.2),
            Text("Output\n(2)", font_size=10).next_to(nodes[3], DOWN, buff=0.2),
        )

        self.play(FadeIn(layer_labels))
        self.wait(0.5)

        # Animacja forward pass
        for i in range(len(layers) - 1):
            self.play(edges[i].animate.set_stroke(color=YELLOW, opacity=0.6, width=1), run_time=0.3)
            self.play(nodes[i + 1].animate.set_color(YELLOW).set_fill(opacity=0.7), run_time=0.3)
            self.play(
                edges[i].animate.set_stroke(color=WHITE, opacity=0.05, width=0.5), run_time=0.2
            )
            if i > 0:
                self.play(nodes[i].animate.set_color(BLUE).set_fill(opacity=0.3), run_time=0.2)

        # Output labels
        pitch_label = (
            Text("Pitch°", font_size=10).next_to(nodes[-1][0], RIGHT, buff=0.3).set_color(GREEN)
        )
        yaw_label = Text("Yaw°", font_size=10).next_to(nodes[-1][1], RIGHT, buff=0.3).set_color(RED)

        self.play(Write(pitch_label), Write(yaw_label), run_time=0.5)
        self.play(nodes[-1].animate.set_color(BLUE).set_fill(opacity=0.3))

        self.wait(2)


class Scene3_AnglesIn3d(ThreeDScene):
    def construct(self):
        # 1. Inicjalizacja osi 3D (Zasada: X, Y = płaszczyzna pozioma, Z = wysokość)
        axes = ThreeDAxes(
            x_range=[-1000, 1000, 500],
            y_range=[-1000, 1000, 500],
            z_range=[0, 1200, 300],
            x_length=10,
            y_length=10,
            z_length=3.2,
            axis_config={"stroke_width": 2},
        )

        # Domyślny widok z końca Scene1
        default_phi = 75 * DEGREES
        default_theta = 105 * DEGREES
        default_zoom = 0.35
        self.set_camera_orientation(phi=default_phi, theta=default_theta, zoom=default_zoom)
        self.play(Create(axes))

        # Pozycje i punkty (dokładnie jak w Scene1)
        shooter_coords = np.array([-600, -600, 100])
        target_coords = np.array([600, 600, 500])

        shooter_pos = axes.c2p(*shooter_coords)
        target_pos = axes.c2p(*target_coords)

        # POPRAWKA KOLORÓW: Shooter = RED, Target = BLUE
        shooter_dot = Dot3D(point=shooter_pos, color=RED, radius=0.08)
        target_dot = Dot3D(point=target_pos, color=BLUE, radius=0.08)

        self.play(FadeIn(shooter_dot), FadeIn(target_dot))

        midpoint = (shooter_pos + target_pos) / 2

        # Zbliżenie na strzelca, aby pokazać układ lokalny kątów
        self.move_camera(frame_center=shooter_pos + np.array([0, 0, 0.2]), zoom=1.2, run_time=1.5)
        self.wait(0.25)

        # Kąty docelowe w radianach (wyliczone idealnie do celu)
        dx = target_coords[0] + 500 - shooter_coords[0]
        dy = target_coords[1] - 700 - shooter_coords[1]
        dz = target_coords[2] + 500 - shooter_coords[2]
        yaw = np.arctan2(dy, dx)
        pitch = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
        yaw_deg = np.degrees(yaw)
        pitch_deg = np.degrees(pitch)

        # TRACKERY (Zadeklarowane razem na początku, aby uniknąć NameError)
        yaw_tracker = ValueTracker(0.0)
        pitch_tracker = ValueTracker(0.0)
        pitch_label_yaw_val = ValueTracker(0.0)

        # Promienie łuków w przestrzeni TF2
        R_yaw_tf2 = 300.0
        R_pitch_tf2 = 300.0

        # --- DYNAMICZNE ŁUKI (JAKO WYPEŁNIONE SEKTORY W PRZESTRZENI TF2) ---
        def get_yaw_polygon():
            val = max(0.001, yaw_tracker.get_value())
            points = [shooter_pos]
            for u in np.linspace(0, val, max(10, int(val * 10))):
                tf2_pt = shooter_coords + np.array(
                    [R_yaw_tf2 * np.cos(u), R_yaw_tf2 * np.sin(u), 0.0]
                )
                points.append(axes.c2p(*tf2_pt))
            return Polygon(*points, color=RED, fill_opacity=0.3, stroke_width=2)

        yaw_arc = always_redraw(get_yaw_polygon)

        def get_pitch_polygon():
            val = max(0.001, pitch_tracker.get_value())
            yaw_val = yaw_tracker.get_value()
            points = [shooter_pos]
            for u in np.linspace(0, val, max(10, int(val * 10))):
                tf2_pt = shooter_coords + np.array(
                    [
                        R_pitch_tf2 * np.cos(u) * np.cos(yaw_val),
                        R_pitch_tf2 * np.cos(u) * np.sin(yaw_val),
                        R_pitch_tf2 * np.sin(u),
                    ]
                )
                points.append(axes.c2p(*tf2_pt))
            return Polygon(*points, color=GREEN, fill_opacity=0.3, stroke_width=2)

        pitch_arc = always_redraw(get_pitch_polygon)

        # --- DYNAMICZNE ETYKIETY (Z USTAWIONĄ ORIENTACJĄ FRONTOWĄ) ---
        yaw_label = always_redraw(
            lambda: Text(
                f"Yaw: {int(np.degrees(yaw_tracker.get_value()))}°", font_size=14, color=RED
            )
            .scale(0.7)
            .move_to(
                axes.c2p(
                    *(
                        shooter_coords
                        + np.array(
                            [
                                (R_yaw_tf2 + 100) * np.cos(yaw_tracker.get_value() / 2),
                                (R_yaw_tf2 + 100) * np.sin(yaw_tracker.get_value() / 2),
                                0.0,
                            ]
                        )
                    )
                )
            )
        )

        pitch_label = always_redraw(
            lambda: Text(
                f"Pitch: {int(np.degrees(pitch_tracker.get_value()))}°", font_size=14, color=GREEN
            )
            .scale(0.7)
            .move_to(
                axes.c2p(
                    *(
                        shooter_coords
                        + np.array(
                            [
                                (R_pitch_tf2 + 100)
                                * np.cos(pitch_tracker.get_value() / 2)
                                * np.cos(pitch_label_yaw_val.get_value()),
                                (R_pitch_tf2 + 100)
                                * np.cos(pitch_tracker.get_value() / 2)
                                * np.sin(pitch_label_yaw_val.get_value()),
                                (R_pitch_tf2 + 100) * np.sin(pitch_tracker.get_value() / 2),
                            ]
                        )
                    )
                )
            )
        )

        # 1) Prezentacja YAW
        self.add_fixed_orientation_mobjects(yaw_label)
        self.add(yaw_arc)
        self.play(yaw_tracker.animate.set_value(TAU), run_time=2.0, rate_func=linear)
        self.wait(0.2)
        self.play(yaw_tracker.animate.set_value(yaw), run_time=1.0)
        self.wait(0.5)

        # Ukrywamy YAW przed pokazaniem PITCH
        self.remove_fixed_orientation_mobjects(yaw_label)
        self.play(FadeOut(yaw_arc), FadeOut(yaw_label), run_time=0.5)

        # 2) Prezentacja PITCH
        self.add_fixed_orientation_mobjects(pitch_label)
        self.add(pitch_arc)
        self.play(pitch_tracker.animate.set_value(PI / 4), run_time=1.5)
        self.wait(0.2)
        self.play(pitch_tracker.animate.set_value(pitch), run_time=1.0)
        self.wait(0.8)

        # Ukrywamy PITCH przed finałowym zestawieniem
        self.remove_fixed_orientation_mobjects(pitch_label)
        self.play(FadeOut(pitch_arc), FadeOut(pitch_label), run_time=0.5)

        # Przywracamy trackery do zera na potrzeby wspólnej animacji wejścia
        yaw_tracker.set_value(0.0)
        pitch_tracker.set_value(0.0)

        # 3) Pokazanie obu łuków i etykiet jednocześnie
        self.add_fixed_orientation_mobjects(yaw_label, pitch_label)
        self.add(yaw_arc, pitch_arc)
        self.play(
            yaw_tracker.animate.set_value(yaw),
            pitch_tracker.animate.set_value(pitch),
            pitch_label_yaw_val.animate.set_value(yaw),
            run_time=1.5,
        )
        self.wait(0.4)

        # Obrót kamery
        self.move_camera(theta=default_theta + 60 * DEGREES, run_time=2.5, rate_func=there_and_back)
        self.wait(0.5)

        # Kalkulacja ostatecznego kierunku strzału
        f_yaw = yaw_tracker.get_value()
        f_pitch = pitch_tracker.get_value()

        dir_vector = np.array(
            [np.cos(f_pitch) * np.cos(f_yaw), np.cos(f_pitch) * np.sin(f_yaw), np.sin(f_pitch)]
        )

        # Droga "hen hen daleko" (poza zakres osi)
        far_coords = shooter_coords + dir_vector * 3000.0
        far_point = axes.c2p(*far_coords)

        # --- MATEMATYKA: WYLICZENIE NAJBLIŻSZEGO PUNKTU DO CELU (w przestrzeni Manima dla wizualnego kąta 90°) ---
        ray_vec_m = far_point - shooter_pos
        ray_dir_m = ray_vec_m / np.linalg.norm(ray_vec_m)

        shooter_to_target_m = target_pos - shooter_pos
        closest_distance_m = np.dot(shooter_to_target_m, ray_dir_m)

        closest_point = shooter_pos + ray_dir_m * closest_distance_m
        closest_coords = np.array(axes.p2c(closest_point))

        full_trajectory = Line(
            start=shooter_pos, end=far_point, color=WHITE, stroke_width=1.5, stroke_opacity=0.4
        )
        bullet_dot = Dot3D(point=shooter_pos, color=YELLOW, radius=0.06)

        # Dodanie wektora kierunkowego z miejsca strzału
        vector_end_coords = shooter_coords + dir_vector * 400.0
        vector_end_pos = axes.c2p(*vector_end_coords)
        vector_arrow = Arrow3D(
            start=shooter_pos,
            end=vector_end_pos,
            color=YELLOW,
        )

        # Pozostawiamy łuki widoczne (zgodnie z życzeniem)
        self.play(Create(full_trajectory), Create(vector_arrow), run_time=0.5)

        # 4) Strzał i jednoczesny ruch kamery
        self.add(bullet_dot)
        self.move_camera(
            frame_center=midpoint,
            zoom=0.75,
            run_time=2.5,
            rate_func=smooth,
            added_anims=[bullet_dot.animate.move_to(closest_point)],
        )

        # Wizualizacja zatrzymania w najbliższym punkcie
        closest_dot = Dot3D(point=closest_point, color=ORANGE, radius=0.06)
        dist_text = Text(f"{0}", font_size=28, color=ORANGE)

        if np.linalg.norm(closest_coords - target_coords) > 1.0:
            dashed_line = DashedLine(
                start=closest_point, end=target_pos, color=ORANGE, stroke_width=2
            )

            v1 = target_pos - closest_point
            v1_unit = v1 / np.linalg.norm(v1)
            v2 = shooter_pos - closest_point
            v2_unit = v2 / np.linalg.norm(v2)

            p1 = closest_point + v1_unit * 0.4
            p2 = closest_point + v1_unit * 0.4 + v2_unit * 0.4
            p3 = closest_point + v2_unit * 0.4

            right_angle = VGroup(
                Line(p1, p2, color=ORANGE, stroke_width=2),
                Line(p2, p3, color=ORANGE, stroke_width=2),
            )

            dist_val = np.linalg.norm(closest_coords - target_coords)
            dist_text = Text(f"{int(dist_val)}", font_size=38, color=ORANGE)

            v_ray_m = closest_point - shooter_pos
            v_target_m = target_pos - shooter_pos
            n_vec = np.cross(v_ray_m, v_target_m)
            if n_vec[2] < 0:
                n_vec = -n_vec
            n_unit = n_vec / np.linalg.norm(n_vec)

            dist_text.move_to((closest_point + target_pos) / 2 + n_unit * 0.3)
            # self.add_fixed_orientation_mobjects(dist_text)

            self.play(FadeIn(closest_dot), Create(dashed_line), Create(right_angle), run_time=1.0)
        else:
            self.play(FadeIn(closest_dot), run_time=1.0)
        self.wait(1)

        # Widok z góry na płaszczyznę utworzoną przez promień i cel (2D projection)
        v_ray = closest_point - shooter_pos
        v_target = target_pos - shooter_pos

        normal_vec = np.cross(v_ray, v_target)
        # Zapewnienie, że patrzymy z "góry" (dodatnie Z)
        if normal_vec[2] < 0:
            normal_vec = -normal_vec

        normal_unit = normal_vec / np.linalg.norm(normal_vec)

        new_phi = np.arccos(normal_unit[2])
        # W Manimie dodanie PI/2 do theta często ustawia oś poziomą w odpowiednim kadrze
        new_theta = np.arctan2(normal_unit[1], normal_unit[0]) + PI / 2

        triangle_center = (shooter_pos + closest_point + target_pos) / 3.0

        self.move_camera(
            phi=new_phi,
            theta=new_theta,
            frame_center=triangle_center,
            zoom=0.45,
            run_time=3.0,
            rate_func=smooth,
        )
        self.play(Write(dist_text), run_time=0.5)
        self.wait(3.5)
