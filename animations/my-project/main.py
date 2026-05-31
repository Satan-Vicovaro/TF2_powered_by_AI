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
            Text("Pitch°", font_size=8).next_to(nodes[-1][0], RIGHT, buff=0.3).set_color(GREEN)
        )
        yaw_label = Text("Yaw°", font_size=8).next_to(nodes[-1][1], RIGHT, buff=0.3).set_color(RED)

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


class Scene4_NeuralNetwork(Scene):
    def construct(self):
        # Tytuł
        main_title = Text("DDPG Architecture", font_size=32)
        main_title.to_edge(UP)
        self.play(Write(main_title))

        def build_network(layer_sizes, input_texts, output_texts, title_text):
            layer_spacing = 2.0
            node_spacing = 0.04
            node_radius = 0.045

            input_color = BLUE
            hidden_color = GRAY
            output_color = RED

            vgroups_layers = []
            for i, num_nodes in enumerate(layer_sizes):
                layer_group = VGroup()
                for j in range(num_nodes):
                    color = (
                        input_color
                        if i == 0
                        else (output_color if i == len(layer_sizes) - 1 else hidden_color)
                    )
                    node = Circle(radius=node_radius, color=color, fill_opacity=1, stroke_width=1)
                    layer_group.add(node)
                layer_group.arrange(DOWN, buff=node_spacing)
                vgroups_layers.append(layer_group)

            nn_group = VGroup(*vgroups_layers)
            nn_group.arrange(RIGHT, buff=layer_spacing)

            input_labels = VGroup()
            for i, text in enumerate(input_texts):
                label = MathTex(text, font_size=18) if "_" in text else Text(text, font_size=14)
                label.next_to(vgroups_layers[0][i], LEFT, buff=0.15)
                input_labels.add(label)

            output_labels = VGroup()
            for i, text in enumerate(output_texts):
                label = Text(text, font_size=16)
                label.next_to(vgroups_layers[-1][i], RIGHT, buff=0.15)
                output_labels.add(label)

            edges = VGroup()
            for i in range(len(vgroups_layers) - 1):
                for node1 in vgroups_layers[i]:
                    for node2 in vgroups_layers[i + 1]:
                        edge = Line(
                            node1.get_right(),
                            node2.get_left(),
                            stroke_width=0.4,
                            stroke_opacity=0.15,
                        )
                        edges.add(edge)

            title = Text(title_text, font_size=20, color=YELLOW)
            title.next_to(nn_group, UP, buff=0.2)

            full_group = VGroup(edges, nn_group, input_labels, output_labels, title)

            return full_group, vgroups_layers, input_labels, output_labels, edges, title

        critic_group, c_layers, c_in_lbl, c_out_lbl, c_edges, c_title = build_network(
            [8, 16, 16, 1],
            ["x_s", "y_s", "z_s", "x_t", "y_t", "z_t", "Pitch", "Yaw"],
            ["Q-Value"],
            "Critic",
        )

        actor_group, a_layers, a_in_lbl, a_out_lbl, a_edges, a_title = build_network(
            [6, 16, 16, 2], ["x_s", "y_s", "z_s", "x_t", "y_t", "z_t"], ["Pitch", "Yaw"], "Actor"
        )

        networks_group = VGroup(critic_group, actor_group).arrange(DOWN, buff=0.4)
        networks_group.to_edge(RIGHT, buff=0.8).shift(DOWN * 0.2)

        # Dane tekstowe na lewej stronie
        import random

        random.seed(0)
        xs, ys, zs = (
            round(random.uniform(-1, 1), 2),
            round(random.uniform(-1, 1), 2),
            round(random.uniform(-1, 1), 2),
        )
        xt, yt, zt = (
            round(random.uniform(-1, 1), 2),
            round(random.uniform(-1, 1), 2),
            round(random.uniform(-1, 1), 2),
        )
        pitch_val, yaw_val = round(random.uniform(-0.5, 0.5), 2), round(
            random.uniform(-0.5, 0.5), 2
        )
        reward_val = round(random.uniform(-10, 10), 1)

        header1 = Text("For:", font_size=20, color=YELLOW)
        state_s = VGroup(
            Tex(f"$x_s$ = {xs}", font_size=20),
            Tex(f"$y_s$ = {ys}", font_size=20),
            Tex(f"$z_s$ = {zs}", font_size=20),
        ).arrange(DOWN, aligned_edge=LEFT)

        state_t = VGroup(
            Tex(f"$x_t$ = {xt}", font_size=20),
            Tex(f"$y_t$ = {yt}", font_size=20),
            Tex(f"$z_t$ = {zt}", font_size=20),
        ).arrange(DOWN, aligned_edge=LEFT)
        state_group = VGroup(state_s, state_t).arrange(RIGHT, buff=0.5)

        header2 = Text("Actor decision:", font_size=20, color=YELLOW)
        action_text = VGroup(
            Tex(f"pitch = {pitch_val}", font_size=20),
            Tex(f"yaw = {yaw_val}", font_size=20),
        ).arrange(DOWN, aligned_edge=LEFT)

        header3 = Text(f"Rewarded: {reward_val}", font_size=20, color=GREEN)

        text_panel = VGroup(header1, state_group, header2, action_text, header3)
        text_panel.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        text_panel.to_edge(LEFT, buff=0.5).shift(DOWN * 0.2)

        # Animacje sekwencyjne
        self.play(
            FadeIn(c_layers[0]),
            Write(c_in_lbl),
            FadeIn(c_title),
            FadeIn(a_layers[0]),
            Write(a_in_lbl),
            FadeIn(a_title),
            run_time=1,
        )
        self.play(
            FadeIn(c_layers[1]),
            FadeIn(c_layers[2]),
            FadeIn(a_layers[1]),
            FadeIn(a_layers[2]),
            run_time=1,
        )
        self.play(
            FadeIn(c_layers[3]), Write(c_out_lbl), FadeIn(a_layers[3]), Write(a_out_lbl), run_time=1
        )

        self.play(Create(c_edges), Create(a_edges), run_time=2)

        # Pokazanie sekcji wejściowej na panelu
        self.play(Write(header1), Write(state_group), run_time=1.5)
        self.wait(0.5)

        # Animacja wartości wejściowych przelatujących do wezłów wejściowych Actora
        state_copies = VGroup(*[m.copy() for m in state_s.submobjects + state_t.submobjects])
        self.play(
            *[
                copy.animate.move_to(a_layers[0][i].get_center()).set_opacity(0).scale(0.5)
                for i, copy in enumerate(state_copies)
            ],
            run_time=1.5
        )

        # Propagacja w przód dla sieci Actor
        self.play(a_edges.animate.set_color(YELLOW).set_opacity(0.6), run_time=0.8)
        self.play(a_edges.animate.set_color(WHITE).set_opacity(0.15), run_time=0.5)

        # Pojawienie się decyzji wyjściowych z sieci Actor
        pitch_out = MathTex(str(pitch_val), font_size=20, color=YELLOW).next_to(a_layers[-1][0], RIGHT, buff=0.8)
        yaw_out = MathTex(str(yaw_val), font_size=20, color=YELLOW).next_to(a_layers[-1][1], RIGHT, buff=0.8)
        self.play(FadeIn(pitch_out), FadeIn(yaw_out), run_time=0.5)
        self.wait(0.5)

        # Nagłówek dla decyzji Actora
        self.play(Write(header2), run_time=0.5)

        # Transformacja wyników w tekst na lewym panelu
        self.play(
            ReplacementTransform(pitch_out, action_text[0]),
            ReplacementTransform(yaw_out, action_text[1]),
            run_time=1.5
        )
        self.wait(0.5)

        # Wypisanie nagrody od środowiska
        self.play(Write(header3), run_time=1.0)

        # --- ANIMACJA DLA CRITIC'A ---
        # Animacja wartości wejściowych dla Critic'a (stany + akcje)
        critic_inputs_copies = VGroup(
            *[m.copy() for m in state_s.submobjects],
            *[m.copy() for m in state_t.submobjects],
            action_text[0].copy(),
            action_text[1].copy()
        )
        
        self.play(
            *[
                copy.animate.move_to(c_layers[0][i].get_center()).set_opacity(0).scale(0.5)
                for i, copy in enumerate(critic_inputs_copies)
            ],
            run_time=1.5
        )

        # Propagacja w przód dla sieci Critic
        self.play(c_edges.animate.set_color(YELLOW).set_opacity(0.6), run_time=0.8)
        self.play(c_edges.animate.set_color(WHITE).set_opacity(0.15), run_time=0.5)

        # Pojawienie się decyzji wyjściowych z sieci Critic (Q-Value)
        q_val = round(reward_val + random.uniform(-0.5, 0.5), 2)
        q_out = MathTex(str(q_val), font_size=20, color=YELLOW).next_to(c_layers[-1][0], RIGHT, buff=1.0)
        self.play(FadeIn(q_out), run_time=0.5)
        self.wait(0.5)

        # Dodanie tekstu Critic na panelu
        header4 = Text("Critic evaluation:", font_size=20, color=YELLOW)
        q_text = Tex(f"Q = {q_val}", font_size=20)
        q_group = VGroup(header4, q_text).arrange(DOWN, aligned_edge=LEFT)
        q_group.next_to(text_panel, DOWN, buff=0.4, aligned_edge=LEFT)

        self.play(Write(header4), run_time=0.5)
        self.play(ReplacementTransform(q_out, q_text), run_time=1.5)

        self.wait(1)

        # --- ANIMACJA POŁĄCZENIA ACTOR I CRITIC ---
        # Przenosimy wszystkie wartości nad sieci neuronowe, ułożone poziomo
        values_bar = VGroup(
            state_s[0], state_s[1], state_s[2],
            state_t[0], state_t[1], state_t[2],
            action_text[0], action_text[1],
            header3,
            q_text
        )
        values_bar.generate_target()
        values_bar.target.arrange(RIGHT, buff=0.4).scale(0.8).next_to(main_title, DOWN, buff=0.2)

        self.play(
            FadeOut(header1),
            FadeOut(header2),
            FadeOut(header4),
            MoveToTarget(values_bar),
            run_time=1.5
        )

        # Przesuwamy sieć Critic na prawą stronę
        self.play(
            critic_group.animate.move_to(RIGHT * 3.2 + DOWN * 0.2),
            run_time=1.5
        )

        # Najpierw ukrywamy etykiety, aby zapobiec nakładaniu się tekstów w trakcie ruchu
        self.play(
            FadeOut(c_in_lbl[6]),
            FadeOut(c_in_lbl[7]),
            FadeOut(a_out_lbl[0]),
            FadeOut(a_out_lbl[1]),
            run_time=0.5
        )

        # Dopasowujemy pozycję sieci Actor tak, aby jej wyjścia idealnie nałożyły się na wejścia Pitch/Yaw Critica
        target_p = c_layers[0][6].get_center()
        shift_vec = target_p - a_layers[-1][0].get_center()
        
        self.play(
            actor_group.animate.shift(shift_vec),
            run_time=1.5
        )

        # Przepływ sygnału pokazujący, że węzły stały się jednym elementem przekazującym dane
        self.play(
            a_layers[-1][0].animate.set_color(RED).set_opacity(0.8),
            a_layers[-1][1].animate.set_color(RED).set_opacity(0.8),
            c_layers[0][6].animate.set_color(RED).set_opacity(0.8),
            c_layers[0][7].animate.set_color(RED).set_opacity(0.8),
            run_time=1.0
        )

        self.wait(3)


class Scene5_NeuralNetwork(Scene):
    def construct(self):
        # Tytuł
        title = Text('"Normal" Network Architecture', font_size=36)
        title.to_edge(UP)
        self.play(Write(title))

        # Konfiguracja Sieci
        layer_spacing = 2.5
        node_spacing = 0.2
        node_radius = 0.1

        input_color = BLUE
        hidden_color = GRAY
        output_color = RED

        # 1. Warstwa wejściowa (4 węzły + kropki + 4 węzły = 9 elementów wizualnych)
        input_layer = VGroup()
        input_nodes = []
        for i in range(4):
            node = Circle(radius=node_radius, color=input_color, fill_opacity=0.1)
            input_layer.add(node)
            input_nodes.append(node)

        dots = MathTex("\\vdots")
        input_layer.add(dots)

        for i in range(4):
            node = Circle(radius=node_radius, color=input_color, fill_opacity=0.1)
            input_layer.add(node)
            input_nodes.append(node)

        input_layer.arrange(DOWN, buff=node_spacing)

        # 2. Ukryte warstwy (16 węzłów każda)
        hidden_layer_1 = VGroup()
        for i in range(16):
            node = Circle(radius=node_radius, color=hidden_color, fill_opacity=0.1)
            hidden_layer_1.add(node)
        hidden_layer_1.arrange(DOWN, buff=node_spacing)

        hidden_layer_2 = VGroup()
        for i in range(16):
            node = Circle(radius=node_radius, color=hidden_color, fill_opacity=0.1)
            hidden_layer_2.add(node)
        hidden_layer_2.arrange(DOWN, buff=node_spacing)

        # 3. Warstwa wyjściowa (10 węzłów)
        output_layer = VGroup()
        for i in range(10):
            node = Circle(radius=node_radius, color=output_color, fill_opacity=0.1)
            output_layer.add(node)
        output_layer.arrange(DOWN, buff=node_spacing)

        # Grupowanie wszystkich warstw
        nn_group = VGroup(input_layer, hidden_layer_1, hidden_layer_2, output_layer)
        nn_group.arrange(RIGHT, buff=layer_spacing)
        nn_group.move_to(LEFT * 1.5)
        nn_group.shift(DOWN * 0.2)

        # Etykieta ilości węzłów wejściowych
        input_desc = Text("100 Nodes", font_size=24, color=BLUE)
        input_desc.next_to(input_layer, DOWN, buff=0.3)

        # Etykiety wyjściowe (0 do 9)
        output_labels = VGroup()
        for i in range(10):
            label = Text(str(i), font_size=20)
            label.next_to(output_layer[i], RIGHT, buff=0.2)
            output_labels.add(label)

        # Krawędzie i słowniki do szybkiego dostępu
        edges = VGroup()
        edges_in_h1 = {}
        for i, node1 in enumerate(input_nodes):
            for j, node2 in enumerate(hidden_layer_1):
                edge = Line(
                    node1.get_right(), node2.get_left(), stroke_width=0.6, stroke_opacity=0.15
                )
                edges.add(edge)
                edges_in_h1[(i, j)] = edge

        edges_h1_h2 = {}
        for i, node1 in enumerate(hidden_layer_1):
            for j, node2 in enumerate(hidden_layer_2):
                edge = Line(
                    node1.get_right(), node2.get_left(), stroke_width=0.6, stroke_opacity=0.15
                )
                edges.add(edge)
                edges_h1_h2[(i, j)] = edge

        edges_h2_out = {}
        for i, node1 in enumerate(hidden_layer_2):
            for j, node2 in enumerate(output_layer):
                edge = Line(
                    node1.get_right(), node2.get_left(), stroke_width=0.6, stroke_opacity=0.15
                )
                edges.add(edge)
                edges_h2_out[(i, j)] = edge

        # Animacje początkowe
        self.play(FadeIn(input_layer), Write(input_desc), run_time=1)
        self.play(FadeIn(hidden_layer_1), FadeIn(hidden_layer_2), run_time=1)
        self.play(FadeIn(output_layer), Write(output_labels), run_time=1)

        self.play(Create(edges), run_time=2)
        self.wait(0.5)

        # Symulacja forward pass (Wypełnianie od dołu)
        import random

        random.seed(42)  # Stały seed dla stabilnej animacji

        def fire_all_nodes(layer, is_output=False):
            anims = []

            activations = [random.uniform(0.1, 0.9) for _ in range(len(layer))]

            if is_output:
                winner_idx = random.randint(0, len(layer) - 1)
                activations[winner_idx] = 1.0

                for i, node in enumerate(layer):
                    val = activations[i]
                    target_color = interpolate_color(node.get_color(), WHITE, val * 0.7)
                    anims.append(node.animate.set_color(target_color).set_fill(opacity=val))

                return anims, winner_idx

            for i, node in enumerate(layer):
                val = activations[i]
                target_color = interpolate_color(node.get_color(), WHITE, val * 0.7)
                anims.append(node.animate.set_color(target_color).set_fill(opacity=val))

            return anims

        def fire_all_edges(edge_dict):
            anims = []
            for edge in edge_dict.values():
                val = random.uniform(0.1, 0.5)
                anims.append(edge.animate.set_opacity(val))
            return anims

        # Krok po kroku przepływ informacji
        # Krok 1: Wejścia
        anims_in = fire_all_nodes(input_nodes)
        self.play(*anims_in, run_time=0.6)
        self.play(*fire_all_edges(edges_in_h1), run_time=0.6)

        # Krok 2: Ukryta warstwa 1
        anims_h1 = fire_all_nodes(hidden_layer_1)
        self.play(*anims_h1, run_time=0.6)
        self.play(*fire_all_edges(edges_h1_h2), run_time=0.6)

        # Krok 3: Ukryta warstwa 2
        anims_h2 = fire_all_nodes(hidden_layer_2)
        self.play(*anims_h2, run_time=0.6)
        self.play(*fire_all_edges(edges_h2_out), run_time=0.6)

        # Krok 4: Wyjście
        out_anims, winner_idx = fire_all_nodes(output_layer, is_output=True)
        self.play(*out_anims, run_time=0.6)

        # Animacja ramki skanującej wyjścia
        scan_box = Square(side_length=0.4, color=YELLOW, stroke_width=3)
        scan_box.move_to(output_layer[0].get_center())

        self.play(Create(scan_box), run_time=0.3)

        # Skanowanie przez wszystkie węzły
        for i in range(1, len(output_layer)):
            self.play(scan_box.animate.move_to(output_layer[i].get_center()), run_time=0.15)

        # Powrót do zwycięskiego węzła i zaznaczenie "max"
        self.play(
            scan_box.animate.move_to(output_layer[winner_idx].get_center()).set_color(RED),
            run_time=0.4,
        )

        max_label = Text("max", font_size=20, color=RED).next_to(
            output_labels[winner_idx], RIGHT, buff=0.2
        )

        self.play(
            output_layer[winner_idx].animate.scale(1.3),
            output_labels[winner_idx].animate.set_color(RED).scale(1.5),
            scan_box.animate.scale(1.3),
            Write(max_label),
            run_time=0.5,
        )
        # Dodanie tablicy wartości obok wyjść (0.0 lub 1.0)
        # Zgodnie z żądaniem: wybrana akcja (1.0) jest inna niż ta z maksymalną wartością (winner_idx)
        import random

        chosen_idx = random.choice([i for i in range(10) if i != winner_idx])

        values_group = VGroup()
        for i in range(10):
            val_text = "1.0" if i == chosen_idx else "0.0"
            color = RED if i == chosen_idx else WHITE
            val_label = Text(val_text, font_size=24, color=color)
            val_label.move_to(output_layer[i].get_center() + RIGHT * 1.5)
            values_group.add(val_label)

        array_title = Text("Correct anwser", font_size=24, color=YELLOW)
        array_title.next_to(values_group, UP, buff=0.4)

        self.play(FadeIn(array_title), FadeIn(values_group), run_time=1)

        # Backpropagation animacja dla wszystkich warstw
        backprop_h2_out = []
        for (i, j), edge in edges_h2_out.items():
            node_opacity = hidden_layer_2[i].get_fill_opacity()
            is_high_brightness = node_opacity > 0.5

            if j == winner_idx and is_high_brightness:
                backprop_h2_out.append(
                    edge.animate.set_color(RED).set_stroke(width=2.0, opacity=0.9)
                )
            elif j == chosen_idx and is_high_brightness:
                backprop_h2_out.append(
                    edge.animate.set_color(GREEN).set_stroke(width=2.0, opacity=0.9)
                )
            else:
                backprop_h2_out.append(
                    edge.animate.set_color(WHITE).set_stroke(width=0.6, opacity=0.15)
                )

        self.play(*backprop_h2_out, run_time=0.6)

        # Propagacja do warstwy h1
        backprop_h1_h2 = []
        for (i, j), edge in edges_h1_h2.items():
            node_opacity = hidden_layer_1[i].get_fill_opacity()
            if node_opacity > 0.5:
                color = random.choice([RED, GREEN])
                backprop_h1_h2.append(
                    edge.animate.set_color(color).set_stroke(width=2.0, opacity=0.9)
                )
            else:
                backprop_h1_h2.append(
                    edge.animate.set_color(WHITE).set_stroke(width=0.6, opacity=0.15)
                )

        self.play(*backprop_h1_h2, run_time=0.6)

        # Propagacja do warstwy wejściowej
        backprop_in_h1 = []
        for (i, j), edge in edges_in_h1.items():
            node_opacity = input_nodes[i].get_fill_opacity()
            if node_opacity > 0.5:
                color = random.choice([RED, GREEN])
                backprop_in_h1.append(
                    edge.animate.set_color(color).set_stroke(width=2.0, opacity=0.9)
                )
            else:
                backprop_in_h1.append(
                    edge.animate.set_color(WHITE).set_stroke(width=0.6, opacity=0.15)
                )

        self.play(*backprop_in_h1, run_time=0.6)

        self.wait(3)
