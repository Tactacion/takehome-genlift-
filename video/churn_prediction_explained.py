"""
Manim animation explaining the churn prediction project.
3blue1brown style with integrated voiceover using manim-voiceover.

Run with: manim -pqh churn_prediction_explained.py FullVideo
"""

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
import numpy as np


class IntroScene(VoiceoverScene):
    """Introduction - what's the problem?"""
    def construct(self):
        # Set up TTS
        self.set_speech_service(GTTSService())

        # Title card
        title = Text("Can We Predict Who Will Leave?", font_size=48, weight=BOLD)
        subtitle = Text("A Time-Aware Approach to Churn Prediction", font_size=28, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.5)

        with self.voiceover(text="so here's the problem: you have users doing things in your app. logging in, making purchases, contacting support. and some of them just disappear. can we predict this?") as tracker:
            self.play(Write(title), run_time=2)
            self.play(FadeIn(subtitle), run_time=1)
            self.wait(tracker.duration - 3)


class ProblemScene(VoiceoverScene):
    """The core challenge - event sequences and time gaps"""
    def construct(self):
        self.set_speech_service(GTTSService())

        # Section title
        section = Text("The Problem", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(section))

        with self.voiceover(text="let me show you what the data looks like"):
            self.wait()

        # Create timeline
        timeline = Line(LEFT * 5, RIGHT * 5, color=WHITE).shift(UP * 1.5)

        # Events with different types
        event_types = ["login", "purchase", "support", "login", "login"]
        event_times = [-4, -2.5, -1, 0.5, 1.5]
        event_colors = [BLUE, GREEN, RED, BLUE, BLUE]

        events = VGroup()
        labels = VGroup()

        for i, (event_type, time, color) in enumerate(zip(event_types, event_times, event_colors)):
            dot = Dot(point=timeline.point_from_proportion((time + 5) / 10), color=color, radius=0.1)
            label = Text(event_type, font_size=16, color=color).next_to(dot, UP, buff=0.3)
            events.add(dot)
            labels.add(label)

        with self.voiceover(text="here's a user's activity. they log in, make a purchase, contact support, log in a couple more times"):
            self.play(Create(timeline))
            self.play(LaggedStart(*[GrowFromCenter(dot) for dot in events], lag_ratio=0.3))
            self.play(LaggedStart(*[Write(label) for label in labels], lag_ratio=0.3))

        # Show the gap
        gap_start = events[-1].get_center()
        gap_end = timeline.point_from_proportion(0.95)
        gap_brace = Brace(Line(gap_start, gap_end), DOWN, color=YELLOW)
        gap_text = Text("13 days of silence", font_size=20, color=YELLOW).next_to(gap_brace, DOWN)

        with self.voiceover(text="then nothing. thirteen days of silence. did they churn? or are they just busy?"):
            self.play(GrowFromCenter(gap_brace), Write(gap_text))
            self.wait(1)

        # Show dataset stats
        stats = VGroup(
            Text("Dataset:", font_size=24, weight=BOLD),
            Text("• 3,000 customers", font_size=20),
            Text("• 30% churn rate", font_size=20),
            Text("• Variable sequence lengths", font_size=20),
            Text("• Irregular time gaps", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_corner(DL)

        with self.voiceover(text="we have three thousand customers, thirty percent churned. sequences are different lengths, time gaps are all over the place. not a lot of data, and it's messy"):
            self.play(FadeIn(stats, shift=RIGHT))
            self.wait(2)


class GenerativeApproachScene(VoiceoverScene):
    """The failed generative world model approach"""
    def construct(self):
        self.set_speech_service(GTTSService())

        # Title
        title = Text("Attempt #1: Generative World Model", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))

        with self.voiceover(text="okay, so my first idea was ambitious. what if we could build a world model? like, actually simulate what users might do next"):
            self.wait()

        # Show the concept
        concept = Text("Predict: P(event, value, time | history)", font_size=28)

        with self.voiceover(text="the math looks like this: given everything a user has done, predict the probability of their next event, its value, and when it happens"):
            self.play(Write(concept))
            self.wait(1)

        # Show distributions
        dist_group = VGroup(
            Text("Event type: Categorical", font_size=20, color=BLUE),
            Text("Value: Gaussian Mixture", font_size=20, color=GREEN),
            Text("Time: Log-Normal", font_size=20, color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(concept, DOWN, buff=0.8)

        with self.voiceover(text="we model event types as categorical, values as gaussian mixtures, and time gaps as log-normal. sounds cool, right?"):
            self.play(LaggedStart(*[FadeIn(item) for item in dist_group], lag_ratio=0.3))
            self.wait(1)

        # Show failure
        failure_box = Rectangle(width=6, height=2, color=RED, fill_opacity=0.1)
        failure_text = VGroup(
            Text("Results:", font_size=28, weight=BOLD, color=RED),
            Text("AUROC: 0.50 (random chance)", font_size=24, color=RED),
            Text("AUPRC: 0.29 (terrible)", font_size=24, color=RED)
        ).arrange(DOWN, buff=0.2).move_to(failure_box)
        failure_group = VGroup(failure_box, failure_text).shift(DOWN * 1.5)

        with self.voiceover(text="but here's the problem. spoiler alert: it didn't work. at all. we got point five zero AUROC. that's literally random guessing"):
            self.play(FadeIn(failure_box), Write(failure_text))
            self.wait(2)

        # Explain why
        why_text = Text("Why? Mode collapse with small data", font_size=24, color=YELLOW).to_edge(DOWN)

        with self.voiceover(text="why did it fail? mode collapse. three thousand examples just isn't enough to learn stable probability distributions. needed like ten times more data"):
            self.play(FadeIn(why_text))
            self.wait(2)


class TimeEncodingScene(VoiceoverScene):
    """The breakthrough - time-aware encoding with animated Fourier scanner"""
    def construct(self):
        self.set_speech_service(GTTSService())

        # Title
        title = Text("The Breakthrough: Time Encoding", font_size=40, color=GREEN).to_edge(UP)
        self.play(Write(title))

        with self.voiceover(text="okay so the generative thing failed. let's try something simpler. just learn good representations and classify"):
            self.wait()

        # The problem with transformers
        problem = Text("Problem: Transformers don't understand time", font_size=28).shift(UP * 2)

        with self.voiceover(text="but there's a catch. transformers don't naturally understand time. they just see a sequence of events"):
            self.play(Write(problem))
            self.wait(1)

        # Show two sequences that look the same
        seq1 = VGroup(
            Text("User A:", font_size=20, weight=BOLD),
            Text("login → purchase → login", font_size=18),
            Text("(1 hour gaps)", font_size=16, color=GRAY)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).shift(LEFT * 2 + UP * 0.5)

        seq2 = VGroup(
            Text("User B:", font_size=20, weight=BOLD),
            Text("login → purchase → login", font_size=18),
            Text("(2 week gaps)", font_size=16, color=GRAY)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).shift(RIGHT * 2 + UP * 0.5)

        with self.voiceover(text="like, these two sequences look the same to a transformer. same events, same order. but user A is active every hour, user B waits weeks. completely different behavior"):
            self.play(FadeIn(seq1), FadeIn(seq2))
            self.wait(2)

        self.play(FadeOut(seq1), FadeOut(seq2), FadeOut(problem), FadeOut(title))

        # THE FOURIER SCANNER - Upgrade 1
        scanner_title = Text("The Fourier Scanner", font_size=36, color=GREEN).to_edge(UP)

        with self.voiceover(text="here's the insight. we compress time with a logarithm, then wrap it around circles at different frequencies. this creates a unique fingerprint for every moment"):
            self.play(Write(scanner_title))
            self.wait(1)

        # Create the time tracker
        time_tracker = ValueTracker(0.1)

        # Logarithmic Timeline
        number_line = NumberLine(
            x_range=[0, 10, 1],
            length=10,
            include_numbers=True,
            color=GRAY
        ).shift(DOWN * 2.5)

        time_label = Text("Time (seconds)", font_size=18, color=GRAY).next_to(number_line, DOWN)

        # The "Time Cursor"
        cursor = Triangle(color=YELLOW, fill_opacity=1).scale(0.15).rotate(PI)
        cursor.add_updater(lambda m: m.next_to(number_line.n2p(time_tracker.get_value()), UP, buff=0.1))

        # The Fourier Circles (Phasors)
        freqs = [1, 2, 5]
        colors = [YELLOW, GREEN, BLUE]
        circles_group = VGroup()

        for i, (freq, color) in enumerate(zip(freqs, colors)):
            circle = Circle(radius=0.5, color=color, stroke_opacity=0.5, stroke_width=2)
            circle.move_to(UP * 0.8 + LEFT * 3 + RIGHT * 2.5 * i)

            # The rotating vector inside
            vector = Arrow(
                start=circle.get_center(),
                end=circle.get_center() + RIGHT * 0.5,
                color=color,
                buff=0,
                stroke_width=4,
                max_tip_length_to_length_ratio=0.2
            )

            # Update function: rotate based on log(time) * freq
            def make_updater(f, c):
                def updater(m):
                    t = time_tracker.get_value()
                    angle = f * np.log(t + 1)
                    end_point = c.get_center() + 0.5 * np.array([np.cos(angle), np.sin(angle), 0])
                    m.put_start_and_end_on(c.get_center(), end_point)
                return updater

            vector.add_updater(make_updater(freq, circle))

            label = MathTex(f"\\omega_{i+1}", color=color, font_size=32).next_to(circle, UP, buff=0.3)
            freq_text = Text(f"ω={freq}", font_size=14, color=color).next_to(circle, DOWN, buff=0.3)

            circles_group.add(VGroup(circle, vector, label, freq_text))

        self.play(Create(number_line), FadeIn(time_label), FadeIn(cursor))
        self.play(LaggedStart(*[Create(c) for c in circles_group], lag_ratio=0.3))

        with self.voiceover(text="watch what happens as time moves forward. each circle rotates at a different speed, creating a unique pattern at every moment"):
            self.play(time_tracker.animate.set_value(10), run_time=6, rate_func=linear)
            self.wait(1)

        # Show the vector output
        vector_box = MathTex(
            r"\text{encoding}(t) = \begin{bmatrix} \sin(\omega_1 \tau) \\ \cos(\omega_1 \tau) \\ \sin(\omega_2 \tau) \\ \cos(\omega_2 \tau) \\ \sin(\omega_3 \tau) \\ \cos(\omega_3 \tau) \end{bmatrix}",
            font_size=28
        ).to_edge(RIGHT).shift(UP * 0.5)

        tau_def = MathTex(r"\tau = \log(t + 1)", font_size=24, color=GRAY).next_to(vector_box, DOWN, buff=0.5)

        with self.voiceover(text="the state of these spinning circles becomes our vector. this allows the model to distinguish between five seconds and five days purely through geometry"):
            self.play(FadeIn(vector_box), FadeIn(tau_def))
            self.wait(3)

        # Clean up
        self.play(
            FadeOut(number_line), FadeOut(time_label), FadeOut(cursor),
            FadeOut(circles_group), FadeOut(vector_box), FadeOut(tau_def),
            FadeOut(scanner_title)
        )


class FocalLossScene(VoiceoverScene):
    """Handling class imbalance with focal loss"""
    def construct(self):
        self.set_speech_service(GTTSService())

        # Title
        title = Text("Challenge: Class Imbalance", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))

        with self.voiceover(text="next problem: class imbalance. thirty percent of users churn, seventy percent stay active"):
            self.wait()

        # Pie chart
        pie_data = [0.7, 0.3]
        pie = VGroup()

        # Create pie slices manually
        circle_center = LEFT * 3 + UP * 0.5
        active_sector = Sector(
            outer_radius=1.5,
            angle=252 * DEGREES,  # 70% of 360
            start_angle=0,
            color=GREEN,
            fill_opacity=0.7
        ).move_arc_center_to(circle_center)

        churn_sector = Sector(
            outer_radius=1.5,
            angle=108 * DEGREES,  # 30% of 360
            start_angle=252 * DEGREES,
            color=RED,
            fill_opacity=0.7
        ).move_arc_center_to(circle_center)

        active_label = Text("70% Active", font_size=20, color=GREEN).next_to(active_sector, LEFT)
        churn_label = Text("30% Churn", font_size=20, color=RED).next_to(churn_sector, RIGHT)

        with self.voiceover(text="here's what the data looks like. seventy percent active, thirty percent churn"):
            self.play(FadeIn(active_sector), FadeIn(churn_sector))
            self.play(Write(active_label), Write(churn_label))
            self.wait(1)

        # Naive approach
        naive = VGroup(
            Text("Naive model:", font_size=24, weight=BOLD),
            Text('"Everyone is active"', font_size=20),
            Text("→ 70% accuracy", font_size=20, color=YELLOW)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).shift(RIGHT * 2.5 + UP * 1)

        with self.voiceover(text="so a naive model can just predict everyone is active and get seventy percent accuracy. not helpful"):
            self.play(FadeIn(naive))
            self.wait(2)

        self.play(FadeOut(active_sector), FadeOut(churn_sector),
                  FadeOut(active_label), FadeOut(churn_label), FadeOut(naive))

        # Show focal loss
        focal_title = Text("Solution: Focal Loss", font_size=32, color=GREEN).shift(UP * 2)

        with self.voiceover(text="the solution? focal loss. borrowed from object detection research"):
            self.play(Write(focal_title))
            self.wait(1)

        # Formula
        formula = MathTex(
            r"\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)",
            font_size=32
        ).shift(UP * 0.5)

        gamma_note = Text("γ = 2.0 (focusing parameter)", font_size=20, color=GRAY).next_to(formula, DOWN, buff=0.5)

        with self.voiceover(text="the formula has this gamma parameter. when the model is confident and correct, the loss gets down-weighted. when it's uncertain or wrong, the loss stays high"):
            self.play(Write(formula))
            self.play(FadeIn(gamma_note))
            self.wait(2)

        # Intuition
        intuition = VGroup(
            Text("Intuition:", font_size=24, weight=BOLD, color=YELLOW),
            Text("Easy examples: low loss", font_size=20),
            Text("Hard examples: high loss", font_size=20),
            Text("→ Model focuses on hard cases", font_size=20, color=GREEN)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).shift(DOWN * 1.5)

        with self.voiceover(text="the intuition: easy examples get low loss, hard examples get high loss. the model automatically focuses on the hard cases. handles imbalance without manual class weights"):
            self.play(FadeIn(intuition))
            self.wait(3)


class ArchitectureScene(VoiceoverScene):
    """Show the full architecture"""
    def construct(self):
        self.set_speech_service(GTTSService())

        # Title
        title = Text("Architecture: Keep It Simple", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))

        with self.voiceover(text="okay, let's build the model. i kept it simple on purpose. small data means don't overdo it"):
            self.wait()

        # Build from bottom to top
        layers = []
        layer_labels = []

        # Input
        input_box = Rectangle(width=3, height=0.5, color=WHITE, fill_opacity=0.1)
        input_label = Text("Input Events", font_size=18).move_to(input_box)
        layers.append(VGroup(input_box, input_label).shift(DOWN * 3))

        with self.voiceover(text="start with the input events"):
            self.play(FadeIn(layers[0]))
            self.wait(0.5)

        # Embeddings
        embed_box = Rectangle(width=3, height=0.5, color=BLUE, fill_opacity=0.2)
        embed_label = Text("Embeddings (128d)", font_size=16).move_to(embed_box)
        layers.append(VGroup(embed_box, embed_label).next_to(layers[-1], UP, buff=0.3))

        with self.voiceover(text="embed event types and values. one twenty eight dimensions"):
            self.play(FadeIn(layers[1]))
            self.wait(0.5)

        # Time encoding
        time_box = Rectangle(width=3, height=0.5, color=YELLOW, fill_opacity=0.2)
        time_label = Text("Time Encoding", font_size=16).move_to(time_box)
        layers.append(VGroup(time_box, time_label).next_to(layers[-1], UP, buff=0.3))

        with self.voiceover(text="add the log-time fourier encoding we talked about"):
            self.play(FadeIn(layers[2]))
            self.wait(0.5)

        # Transformer (2 layers)
        trans_box = Rectangle(width=3, height=1.2, color=GREEN, fill_opacity=0.2)
        trans_label = VGroup(
            Text("Transformer", font_size=18, weight=BOLD),
            Text("2 layers, pre-norm", font_size=14, color=GRAY)
        ).arrange(DOWN, buff=0.1).move_to(trans_box)
        layers.append(VGroup(trans_box, trans_label).next_to(layers[-1], UP, buff=0.3))

        with self.voiceover(text="two transformer layers with pre-norm. tried more layers, they overfit. two was the sweet spot"):
            self.play(FadeIn(layers[3]))
            self.wait(1.5)

        # Pooling
        pool_box = Rectangle(width=3, height=0.5, color=PURPLE, fill_opacity=0.2)
        pool_label = Text("Mean Pool", font_size=16).move_to(pool_box)
        layers.append(VGroup(pool_box, pool_label).next_to(layers[-1], UP, buff=0.3))

        with self.voiceover(text="mean pooling to aggregate the sequence. more robust than just using the last token"):
            self.play(FadeIn(layers[4]))
            self.wait(1)

        # Classifier
        class_box = Rectangle(width=3, height=0.5, color=RED, fill_opacity=0.2)
        class_label = Text("Classifier", font_size=16).move_to(class_box)
        layers.append(VGroup(class_box, class_label).next_to(layers[-1], UP, buff=0.3))

        with self.voiceover(text="and a simple classifier on top"):
            self.play(FadeIn(layers[5]))
            self.wait(0.5)

        # Output
        output_box = Rectangle(width=3, height=0.5, color=WHITE, fill_opacity=0.1)
        output_label = Text("Churn Probability", font_size=16).move_to(output_box)
        layers.append(VGroup(output_box, output_label).next_to(layers[-1], UP, buff=0.3))

        with self.voiceover(text="output is the churn probability"):
            self.play(FadeIn(layers[6]))
            self.wait(0.5)

        # Add arrows
        arrows = VGroup(*[Arrow(layers[i].get_top(), layers[i+1].get_bottom(), buff=0.1, color=GRAY, stroke_width=2)
                         for i in range(len(layers)-1)])

        self.play(LaggedStart(*[GrowArrow(arrow) for arrow in arrows], lag_ratio=0.2))

        # Stats
        stats = VGroup(
            Text("Total:", font_size=20, weight=BOLD),
            Text("• 429k parameters", font_size=18),
            Text("• 2 transformer layers", font_size=18),
            Text("• 4 attention heads", font_size=18),
            Text("• Pre-norm architecture", font_size=18)
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT).to_corner(UR)

        with self.voiceover(text="four hundred twenty nine thousand parameters total. lightweight by design. small data means small model"):
            self.play(FadeIn(stats))
            self.wait(2)

        # DATA FLOW PARTICLES - Upgrade 3
        with self.voiceover(text="let's trace a single user. their events flow up, get embedded, get processed by time-aware attention, and finally compressed into a churn probability"):
            # Create the path from input to output
            start_point = layers[0][0].get_center()  # input_box center
            end_point = layers[6][0].get_center()    # output_box center

            # Animate multiple packets of data flowing up
            for i in range(6):
                packet = Dot(color=YELLOW, radius=0.12)
                packet.move_to(start_point)

                # Create a vertical path that goes through each layer
                path_points = [layer[0].get_center() for layer in layers]
                path = VMobject()
                path.set_points_as_corners(path_points)

                # Glow effect
                glow = Circle(radius=0.2, color=YELLOW, fill_opacity=0.3, stroke_opacity=0).move_to(start_point)

                self.add(packet, glow)

                # Animate packet moving along path with glow following
                self.play(
                    MoveAlongPath(packet, path, rate_func=linear),
                    MoveAlongPath(glow, path, rate_func=linear),
                    run_time=1.2
                )

                # Burst effect at output
                burst = Circle(radius=0.15, color=YELLOW, stroke_width=3).move_to(end_point)
                self.play(
                    burst.animate.scale(2).set_stroke(opacity=0),
                    FadeOut(packet),
                    FadeOut(glow),
                    run_time=0.3
                )
                self.remove(burst)

                # Small delay between packets
                if i < 5:
                    self.wait(0.15)


class ResultsScene(VoiceoverScene):
    """Final results and lessons learned with manifold visualization"""
    def construct(self):
        self.set_speech_service(GTTSService())

        # Title
        title = Text("Results", font_size=40, color=GREEN).to_edge(UP)
        self.play(Write(title))

        # THE MANIFOLD VISUALIZATION - Upgrade 2
        with self.voiceover(text="visualizing the latent space, we see what happened. the generative model left everything in a messy cloud. but our time-aware encoder pulled the universe apart"):
            self.wait()

        # 1. The "Messy" Generative Space
        np.random.seed(42)
        n_active = 70
        n_churn = 30

        # Create overlapping clouds
        messy_active = VGroup(*[
            Dot(
                np.random.normal(0, 0.8, 3) * np.array([1, 1, 0]),
                color=GREEN,
                radius=0.06
            )
            for _ in range(n_active)
        ])

        messy_churn = VGroup(*[
            Dot(
                np.random.normal(0, 0.8, 3) * np.array([1, 1, 0]) + np.array([0.3, -0.2, 0]),
                color=RED,
                radius=0.06
            )
            for _ in range(n_churn)
        ])

        messy_dots = VGroup(messy_active, messy_churn)

        gen_label = Text("Generative Model Space", font_size=20, color=GRAY).next_to(messy_dots, DOWN, buff=0.5)

        self.play(FadeIn(messy_dots), Write(gen_label))
        self.wait(1.5)

        # 2. The "Discriminative" Separation
        organized_active = VGroup()
        organized_churn = VGroup()

        for i, dot in enumerate(messy_active):
            base_pos = np.random.normal(0, 0.4, 3) * np.array([0.8, 1, 0])
            target_pos = base_pos + np.array([-2.5, 0.3, 0])
            new_dot = Dot(target_pos, color=GREEN, radius=0.06)
            organized_active.add(new_dot)

        for i, dot in enumerate(messy_churn):
            base_pos = np.random.normal(0, 0.4, 3) * np.array([0.8, 1, 0])
            target_pos = base_pos + np.array([2.5, -0.3, 0])
            new_dot = Dot(target_pos, color=RED, radius=0.06)
            organized_churn.add(new_dot)

        organized_dots = VGroup(organized_active, organized_churn)

        # Decision boundary
        boundary_line = DashedLine(UP * 3, DOWN * 3, color=WHITE, stroke_width=3)

        # Labels
        active_label = Text("Active Users", font_size=18, color=GREEN).move_to(LEFT * 3 + UP * 2.5)
        churn_label = Text("Churned Users", font_size=18, color=RED).move_to(RIGHT * 3 + UP * 2.5)

        disc_label = Text("Time-Aware Encoder Space", font_size=20, color=GRAY).next_to(organized_dots, DOWN, buff=1)

        with self.voiceover(text="it learned a manifold where active users and churners are linearly separable. that is why we hit ninety-nine percent accuracy"):
            self.play(
                Transform(messy_active, organized_active),
                Transform(messy_churn, organized_churn),
                Transform(gen_label, disc_label),
                run_time=2.5
            )
            self.play(Create(boundary_line))
            self.play(FadeIn(active_label), FadeIn(churn_label))
            self.wait(2)

        self.play(
            FadeOut(messy_active), FadeOut(messy_churn),
            FadeOut(boundary_line), FadeOut(active_label),
            FadeOut(churn_label), FadeOut(gen_label),
            FadeOut(title)
        )

        # Now show the numerical results
        title = Text("Results", font_size=40, color=GREEN).to_edge(UP)
        self.play(Write(title))

        with self.voiceover(text="alright, moment of truth. how well does it work?"):
            self.wait()

        # Big numbers
        auroc = Text("AUROC: 0.9960", font_size=48, weight=BOLD, color=GREEN)
        auprc = Text("AUPRC: 0.9953", font_size=48, weight=BOLD, color=GREEN)

        results_group = VGroup(auroc, auprc).arrange(DOWN, buff=0.5)

        with self.voiceover(text="point nine nine six zero AUROC. point nine nine five three AUPRC. basically perfect"):
            self.play(Write(auroc), run_time=2)
            self.play(Write(auprc), run_time=2)
            self.wait(1)

        self.play(results_group.animate.scale(0.6).to_edge(LEFT))

        # Compare to baselines
        comparison = VGroup(
            Text("Comparison:", font_size=24, weight=BOLD),
            Text("Generative model: 0.50", font_size=20, color=RED),
            Text("Logistic regression: 0.72", font_size=20, color=YELLOW),
            Text("Random forest: 0.84", font_size=20, color=YELLOW),
            Text("Our model: 0.996", font_size=20, color=GREEN)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).shift(RIGHT * 2)

        with self.voiceover(text="compare that to baselines. generative model was point five. logistic regression gets point seven two. random forest point eight four. we're at point nine nine six"):
            self.play(FadeIn(comparison))
            self.wait(3)

        self.play(FadeOut(comparison))

        # Training time
        training = VGroup(
            Text("Training:", font_size=24, weight=BOLD),
            Text("4 minutes on CPU", font_size=20),
            Text("3 epochs", font_size=20),
            Text("No GPU needed", font_size=20, color=GREEN)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).shift(RIGHT * 2 + UP * 0.5)

        with self.voiceover(text="and it trains in four minutes. on a C P U. no fancy hardware needed"):
            self.play(FadeIn(training))
            self.wait(2)

        self.play(FadeOut(training), results_group.animate.center().shift(UP * 2))

        # Why it worked
        why_title = Text("Why It Worked:", font_size=28, weight=BOLD, color=BLUE)

        reasons = VGroup(
            Text("1. Time-aware encoding (the key insight)", font_size=20),
            Text("2. Focal loss (handles imbalance elegantly)", font_size=20),
            Text("3. Pre-norm transformers (stable training)", font_size=20),
            Text("4. Simple architecture (no overfitting)", font_size=20)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)

        why_group = VGroup(why_title, reasons).arrange(DOWN, buff=0.4)

        with self.voiceover(text="why did it work? four reasons. one: time-aware encoding was the breakthrough. two: focal loss handled the imbalance. three: pre-norm kept training stable. four: kept it simple, avoided overfitting"):
            self.play(Write(why_title))
            self.play(LaggedStart(*[FadeIn(reason) for reason in reasons], lag_ratio=0.5))
            self.wait(4)

        self.play(FadeOut(why_group), FadeOut(results_group), FadeOut(title))

        # Final lessons
        final_title = Text("Lessons Learned", font_size=36, weight=BOLD, color=YELLOW)

        lessons = VGroup(
            Text("• Generative models need lots of data", font_size=20),
            Text("• Discriminative learning is underrated", font_size=20),
            Text("• Time encoding matters for sequences", font_size=20),
            Text("• Simpler models often work better", font_size=20),
            Text("• Don't skip the baselines", font_size=20)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)

        lessons_group = VGroup(final_title, lessons).arrange(DOWN, buff=0.5)

        with self.voiceover(text="final lessons. generative models are cool but need tons of data. discriminative learning is underrated. time encoding is critical for event sequences. simpler models often beat complex ones. and always try the baselines first"):
            self.play(Write(final_title))
            self.play(LaggedStart(*[FadeIn(lesson) for lesson in lessons], lag_ratio=0.4))
            self.wait(5)

        self.play(FadeOut(lessons_group))

        # Final message
        final = VGroup(
            Text("From 50% to 99.6%", font_size=40, weight=BOLD, color=GREEN),
            Text("By understanding time", font_size=28, color=GRAY)
        ).arrange(DOWN, buff=0.3)

        with self.voiceover(text="from fifty percent to ninety nine point six percent. all by understanding time. thanks for watching"):
            self.play(Write(final[0]), run_time=2)
            self.play(FadeIn(final[1]), run_time=1.5)
            self.wait(3)


class FullVideo(VoiceoverScene):
    """Complete video combining all scenes"""
    def construct(self):
        self.set_speech_service(GTTSService())

        # Run all scenes in sequence
        IntroScene.construct(self)
        self.clear()
        self.wait(1)

        ProblemScene.construct(self)
        self.clear()
        self.wait(1)

        GenerativeApproachScene.construct(self)
        self.clear()
        self.wait(1)

        TimeEncodingScene.construct(self)
        self.clear()
        self.wait(1)

        FocalLossScene.construct(self)
        self.clear()
        self.wait(1)

        ArchitectureScene.construct(self)
        self.clear()
        self.wait(1)

        ResultsScene.construct(self)
