import numpy as np
from datetime import timedelta, date
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact_manual
from dataclasses import dataclass
import pandas as pd
import os
from IPython.display import display
import matplotlib.gridspec as gridspec
from ipywidgets import interact, HBox, VBox
from scipy import stats
from scipy.stats import norm


def sample_means(data, sample_size):
    means = []

    for _ in range(10_000):
        sample = np.random.choice(data, size=sample_size)
        means.append(np.mean(sample))

    return np.array(means)


def gaussian_clt():
    def _plot(mu, sigma, sample_size):
        #         mu = 10
        #         sigma = 5

        gaussian_population = np.random.normal(mu, sigma, 100_000)
        gaussiam_sample_means = sample_means(gaussian_population, sample_size)
        x_range = np.linspace(
            min(gaussiam_sample_means), max(gaussiam_sample_means), 100
        )

        sample_means_mean = np.mean(gaussiam_sample_means)
        sample_means_std = np.std(gaussiam_sample_means)
        clt_std = sigma / np.sqrt(sample_size)

        estimated_pop_sigma = sample_means_std * np.sqrt(sample_size)

        std_err = abs(clt_std - sample_means_std) / clt_std

        clt_holds = True if std_err < 0.1 else False

        #         print(f"Mean of sample means: {sample_means_mean:.2f}\n")
        #         print(f"Std of sample means: {sample_means_std:.2f}\n")
        #         print(f"Theoretical sigma: {clt_std:.2f}\n")
        #         print(f"Estimated population sigma: {estimated_pop_sigma:.2f}\n")

        #         print(f"Error: {std_err:.2f}\n")
        #         print(f"CLT holds?: {clt_holds}\n")

        mu2 = mu
        sigma2 = sigma / np.sqrt(sample_size)
        #         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 6))
        fig, axes = plt.subplot_mosaic(
            [["top row", "top row"], ["bottom left", "bottom right"]], figsize=(10, 5)
        )

        ax1 = axes["top row"]
        ax2 = axes["bottom left"]
        ax3 = axes["bottom right"]
        sns.histplot(gaussian_population, stat="density", ax=ax1)
        ax1.set_title("Population Distribution")
        ax2.set_title("Sample Means Distribution")
        ax3.set_title("QQ Plot of Sample Means")

        sns.histplot(gaussiam_sample_means, stat="density", ax=ax2, label="hist")
        sns.kdeplot(
            data=gaussiam_sample_means,
            color="crimson",
            ax=ax2,
            label="kde",
            linestyle="dashed",
            fill=True,
        )
        ax2.plot(
            x_range,
            norm.pdf(x_range, loc=mu2, scale=sigma2),
            color="black",
            label="gaussian",
            linestyle="solid",
        )
        ax2.legend()

        stats.probplot(gaussiam_sample_means, plot=ax3, fit=True)
        plt.tight_layout()
        plt.show()

    mu_selection = widgets.FloatSlider(
        value=10.0,
        min=0.01,
        max=50.0,
        step=1.0,
        description="mu",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
    )

    sigma_selection = widgets.FloatSlider(
        value=5.0,
        min=0.01,
        max=20.0,
        step=0.1,
        description="sigma",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
    )

    sample_size_selection = widgets.IntSlider(
        value=2,
        min=2,
        max=100,
        step=1,
        description="sample_size",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )

    interact_manual(
        _plot, sample_size=sample_size_selection, mu=mu_selection, sigma=sigma_selection
    )


def binomial_clt():
    def _plot(n, p, sample_size):
        mu = n * p
        sigma = np.sqrt(n * p * (1 - p)) / np.sqrt(sample_size)
        N = n * sample_size
        #         sigma = np.sqrt(n * p * (1 - p)) / np.sqrt(N)

        binomial_population = np.random.binomial(n, p, 100_000)

        binomial_sample_means = sample_means(binomial_population, sample_size)

        x_range = np.linspace(
            min(binomial_sample_means), max(binomial_sample_means), 100
        )

        condition_val = np.min([N * p, N * (1 - p)])

        condition = True if condition_val >= 5 else False

        sample_means_mean = np.mean(binomial_sample_means)
        sample_means_std = np.std(binomial_sample_means)
        clt_std = np.std(binomial_population) / np.sqrt(sample_size)

        estimated_pop_sigma = sample_means_std * np.sqrt(sample_size)

        std_err = abs(clt_std - sample_means_std) / clt_std

        clt_holds = True if std_err < 0.1 else False

        #         print(f"Value of N: {N}\n")
        print(f"Condition value: {condition_val:.1f}")

        #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig, axes = plt.subplot_mosaic(
            [["top row", "top row"], ["bottom left", "bottom right"]], figsize=(10, 5)
        )

        ax1 = axes["top row"]
        ax2 = axes["bottom left"]
        ax3 = axes["bottom right"]
        ax1.set_title("Population Distribution")
        ax2.set_title("Sample Means Distribution")
        ax3.set_title("QQ Plot of Sample Means")
        sns.histplot(binomial_population, stat="density", ax=ax1)

        sns.histplot(binomial_sample_means, stat="density", ax=ax2, label="hist")
        sns.kdeplot(
            data=binomial_sample_means,
            color="crimson",
            ax=ax2,
            label="kde",
            linestyle="dashed",
            fill=True,
        )
        ax2.plot(
            x_range,
            norm.pdf(x_range, loc=mu, scale=sigma),
            color="black",
            label="gaussian",
            linestyle="solid",
        )
        ax2.legend()
        stats.probplot(binomial_sample_means, plot=ax3, fit=True)
        plt.tight_layout()
        plt.show()

    #         print(f"Condition holds?: {condition} with value of {condition_val:.2f}\n")

    #         print(f"Mean of sample means: {sample_means_mean:.2f}\n")
    #         print(f"Std of sample means: {sample_means_std:.2f}\n")
    #         print(f"Theoretical sigma: {clt_std:.2f}\n")
    #         print(f"Estimated population sigma: {estimated_pop_sigma:.2f}\n")

    sample_size_selection = widgets.IntSlider(
        value=2,
        min=2,
        max=50,
        step=1,
        description="sample_size",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )

    n_selection = widgets.IntSlider(
        value=2,
        min=2,
        max=50,
        step=1,
        description="n",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )

    prob_success_selection = widgets.FloatSlider(
        value=0.5,
        min=0.01,
        max=0.99,
        step=0.1,
        description="p",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
    )

    interact_manual(
        _plot,
        sample_size=sample_size_selection,
        p=prob_success_selection,
        n=n_selection,
    )


def poisson_clt():
    def _plot(mu, sample_size):
        sigma = np.sqrt(mu) / np.sqrt(sample_size)

        poisson_population = np.random.poisson(mu, 100_000)

        poisson_sample_means = sample_means(poisson_population, sample_size)

        x_range = np.linspace(min(poisson_sample_means), max(poisson_sample_means), 100)

        fig, axes = plt.subplot_mosaic(
            [["top row", "top row"], ["bottom left", "bottom right"]], figsize=(10, 5)
        )

        ax1 = axes["top row"]
        ax2 = axes["bottom left"]
        ax3 = axes["bottom right"]
        ax1.set_title("Population Distribution")
        ax2.set_title("Sample Means Distribution")
        ax3.set_title("QQ Plot of Sample Means")
        sns.histplot(poisson_population, stat="density", ax=ax1)

        sns.histplot(poisson_sample_means, stat="density", ax=ax2, label="hist")
        sns.kdeplot(
            data=poisson_sample_means,
            color="crimson",
            ax=ax2,
            label="kde",
            linestyle="dashed",
            fill=True,
        )
        ax2.plot(
            x_range,
            norm.pdf(x_range, loc=mu, scale=sigma),
            color="black",
            label="gaussian",
            linestyle="solid",
        )
        ax2.legend()
        stats.probplot(poisson_sample_means, plot=ax3, fit=True)
        plt.tight_layout()
        plt.show()

    sample_size_selection = widgets.IntSlider(
        value=2,
        min=2,
        max=50,
        step=1,
        description="sample_size",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )

    mu_selection = widgets.FloatSlider(
        value=1.5,
        min=0.01,
        max=5.0,
        #         step=1.0,
        description="mu",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
    )

    interact_manual(_plot, sample_size=sample_size_selection, mu=mu_selection)


def plot_kde_and_qq(sample_means_data, mu_sample_means, sigma_sample_means):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Define the x-range for the Gaussian curve (this is just for plotting purposes)
    x_range = np.linspace(min(sample_means_data), max(sample_means_data), 100)

    # Histogram of sample means (blue)
    sns.histplot(sample_means_data, stat="density", label="hist", ax=ax1)

    # Estimated PDF of sample means (red)
    sns.kdeplot(
        data=sample_means_data,
        color="crimson",
        label="kde",
        linestyle="dashed",
        fill=True,
        ax=ax1,
    )

    # Gaussian curve with estimated mu and sigma (black)
    ax1.plot(
        x_range,
        norm.pdf(x_range, loc=mu_sample_means, scale=sigma_sample_means),
        color="black",
        label="gaussian",
    )

    res = stats.probplot(sample_means_data, plot=ax2, fit=True)

    ax1.legend()
    plt.show()


class your_bday:
    def __init__(self) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        self.fig = fig
        self.ax = ax1
        self.ax_hist = ax2
        self.dates = [
            (date(2015, 1, 1) + timedelta(days=n)).strftime("%m-%d") for n in range(365)
        ]
        self.match = False
        self.bday_str = None
        self.bday_index = None
        self.n_students = 0
        self.history = []
        self.bday_picker = widgets.DatePicker(
            description="Pick your bday",
            disabled=False,
            style={"description_width": "initial"},
        )
        self.start_button = widgets.Button(description="Simulate!")

        display(self.bday_picker)
        display(self.start_button)

        self.start_button.on_click(self.on_button_clicked)

    def on_button_clicked(self, b):
        self.match = False
        self.n_students = 0

        self.get_bday()
        self.add_students()

    def get_bday(self):
        try:
            self.bday_str = self.bday_picker.value.strftime("%m-%d")
        except AttributeError:
            self.ax.set_title(f"Input a valid date and try again!")
            return
        self.bday_index = self.dates.index(self.bday_str)

    def generate_bday(self):
        # gen_bdays = np.random.randint(0, 365, (n_people))
        gen_bday = np.random.randint(0, 365)
        # if not np.isnan(self.y[gen_bday]):
        if gen_bday == self.bday_index:
            self.match = True

    def add_students(self):
        if not self.bday_str:
            return

        while True:
            if self.match:
                self.history.append(self.n_students)
                #                 print(f"Match found. It took {self.n_students} students to get a match")
                n_runs = [i for i in range(len(self.history))]
                self.ax.scatter(n_runs, self.history)
                # counts, bins = np.histogram(self.history)
                # plt.stairs(counts, bins)
                # self.ax_hist.hist(bins[:-1], bins, weights=counts)
                self.ax_hist.clear()
                sns.histplot(data=self.history, ax=self.ax_hist, bins=16)
                # plt.show()
                break

            self.generate_bday()
            self.n_students += 1
            self.ax.set_title(
                f"Match found. It took {self.n_students} students.\nNumber of runs: {len(self.history)+1}"
            )
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()


big_classroom_sizes = [*range(1, 1000, 5)]
small_classroom_sizes = [*range(1, 80)]


def plot_simulated_probs(sim_probs, class_size):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    #     ax.scatter(class_size, sim_probs)
    sns.scatterplot(x=class_size, y=sim_probs, ax=ax, label="simulated probabilities")
    ax.set_ylabel("Simulated Probability")
    ax.set_xlabel("Classroom Size")
    ax.set_title("Probability vs Number of Students")
    ax.plot([0, max(class_size)], [0.5, 0.5], color="red", label="p = 0.5")
    ax.grid(which="minor", color="#EEEEEE", linewidth=0.8)
    ax.minorticks_on()
    ax.legend()
    plt.show()


class third_bday_problem:
    def __init__(self) -> None:
        fig, axes = plt.subplot_mosaic(
            [["top row", "top row"], ["bottom left", "bottom right"]], figsize=(10, 8)
        )
        self.fig = fig
        self.ax = axes["top row"]
        self.count_ax = axes["bottom left"]
        self.ax_hist = axes["bottom right"]
        self.ax.spines["top"].set_color("none")
        self.ax.spines["right"].set_color("none")
        self.ax.spines["left"].set_color("none")
        self.ax.get_yaxis().set_visible(False)
        x = np.arange(365)
        y = np.zeros((365,))
        y[y == 0] = np.nan

        y_match = np.zeros((365,))
        y_match[y_match == 0] = np.nan

        self.x = x
        self.y = y
        self.y_match = y_match
        self.match = False
        self.n_students = 0

        self.dates = [
            (date(2015, 1, 1) + timedelta(days=n)).strftime("%m-%d") for n in range(365)
        ]
        self.month_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        self.history = []
        self.match_index = None
        self.match_str = None

        self.cpoint = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_button_clicked
        )

        # self.start_button = widgets.Button(description="Simulate!")

        # display(self.start_button)

        # self.start_button.on_click(self.on_button_clicked)

    def generate_bday(self):
        gen_bday = np.random.randint(0, 365)

        if not np.isnan(self.y[gen_bday]):
            self.match_index = gen_bday
            self.match_str = self.dates[gen_bday]
            self.y_match[gen_bday] = 1
            self.match = True

        self.y[gen_bday] = 0.5

    def on_button_clicked(self, event):
        if event.inaxes in [self.ax]:
            self.new_run()
            self.add_students()

    def add_students(self):
        while True:
            if self.match:
                self.history.append(self.n_students)
                n_runs = [i for i in range(len(self.history))]
                self.count_ax.scatter(n_runs, self.history)
                self.count_ax.set_ylabel("# of students")
                self.count_ax.set_xlabel("# of simulations")

                month_str = self.month_names[int(self.match_str.split("-")[0]) - 1]
                day_value = self.match_str.split("-")[1]
                self.ax.set_title(
                    f"Match found for {month_str} {day_value}\nIt took {self.n_students} students to get a match"
                )
                self.ax_hist.clear()
                sns.histplot(data=self.history, ax=self.ax_hist, bins="auto")
                break

            self.generate_bday()
            self.n_students += 1
            self.ax.set_title(f"Number of students: {self.n_students}")

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            if not np.isnan(self.y_match).all():
                markerline, stemlines, baseline = self.ax.stem(
                    self.x, self.y_match, markerfmt="*"
                )
                plt.setp(markerline, color="green")
                plt.setp(stemlines, "color", plt.getp(markerline, "color"))
                plt.setp(stemlines, "linestyle", "dotted")
            self.ax.stem(self.x, self.y, markerfmt="o")

    def new_run(self):
        y = np.zeros((365,))
        y[y == 0] = np.nan
        y_match = np.zeros((365,))
        y_match[y_match == 0] = np.nan
        self.y_match = y_match
        self.y = y
        self.n_students = 0
        self.match = False
        self.ax.clear()


class monty_hall_game:
    def __init__(self) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig = fig
        self.ax = ax1
        self.results_ax = ax2
        self.memory_wins = {"switch": 0, "stay": 0}
        self.memory_games = {"switch": 0, "stay": 0}
        self.games_finished = 0
        self.start()

        self.cpoint = self.fig.canvas.mpl_connect("button_press_event", self.click_plot)

    def start(self) -> None:
        self.ax.clear()
        values = [10, 10, 10]
        door_numbers = ["Door 1", "Door 2", "Door 3"]

        self.ax.spines["top"].set_color("none")
        self.ax.spines["right"].set_color("none")
        self.ax.spines["left"].set_color("none")
        self.ax.get_yaxis().set_visible(False)

        self.ax.bar(
            door_numbers,
            values,
            color=["brown", "brown", "brown"],
            width=0.6,
            edgecolor=["black", "black", "black"],
        )
        self.ax.set_title(f"New game started, pick any door.")

        self.prize_coordinates = [-0.15, 0.85, 1.85]

        self.doors, self.winner_index = self.init_monty_hall()
        self.prizes = list(map(lambda x: "GOAT" if x == 0 else "CAR", list(self.doors)))

        self.choice = None
        self.switch = None
        self.temptative_final_door = None
        self.final_choice = None
        self.first_pick = True
        self.game_over = False
        self.won = None
        self.ilegal_move = False

    def click_plot(self, event):
        if event.inaxes in [self.ax]:
            if self.game_over:
                self.start()
                return

            # if self.choice and self.final_choice:
            #     self.start()

            if self.first_pick:
                self.first_pick_mtd(event.xdata)
            else:
                self.second_pick_mtd(event.xdata)

            if ((self.choice is not None) or (self.final_choice is not None)) and (
                not self.ilegal_move
            ):
                self.update_bar_chart()

    def first_pick_mtd(self, x_coord):
        if (x_coord >= -0.3) and (x_coord <= 0.3):
            self.choice = 0
        elif (x_coord >= 0.7) and (x_coord <= 1.3):
            self.choice = 1
        elif (x_coord >= 1.7) and (x_coord <= 2.3):
            self.choice = 2
        else:
            self.choice = None
            #             print("click a door")
            # self.ax.set_title(f"Click on a door to move forward")
            self.start()

    def second_pick_mtd(self, x_coord):
        if (x_coord >= -0.3) and (x_coord <= 0.3):
            self.final_choice = 0
        elif (x_coord >= 0.7) and (x_coord <= 1.3):
            self.final_choice = 1
        elif (x_coord >= 1.7) and (x_coord <= 2.3):
            self.final_choice = 2
        else:
            self.final_choice = None
            #             print("click a door")
            # self.ax.set_title(f"Click on a door to move forward")
            self.start()

        if self.final_choice == self.opened_door:
            self.ax.set_title(f"You selected the opened door.\nThis game doesn't count")
            self.ilegal_move = True
            self.game_over = True

    def update_bar_chart(self):
        if self.first_pick:
            values = [10, 10, 10]
            colors = ["brown", "brown", "brown"]
            edge_colors = ["black", "black", "black"]
            linewidths = [1, 1, 1]
            # colors[self.choice] = "red"
            edge_colors[self.choice] = "red"
            linewidths[self.choice] = 5

            door_numbers = ["Door 1", "Door 2", "Door 3"]
            self.opened_door = self.open_door()
            # values[opened_door] = 0

            colors[self.opened_door] = "gray"

            self.ax.clear()
            self.ax.text(
                self.prize_coordinates[self.opened_door],
                5,
                f"{self.prizes[self.opened_door]}",
            )

            # self.ax.text(0,10,f"You chose door {self.choice} and host opened door {opened_door}")
            # self.ax.text(0.4,5,f"{self.doors}")
            # self.results_ax.bar(door_numbers, values, color = colors,width = 0.6, edgecolor = ['black', 'black', 'black'])

            self.ax.bar(
                door_numbers,
                values,
                color=colors,
                width=0.6,
                edgecolor=edge_colors,
                linewidth=linewidths,
            )
            self.ax.set_title(
                f"You chose door {self.choice+1} and host opened door {self.opened_door+1}.\nDecide your final door."
            )
            self.first_pick = False
        else:
            values = [10, 10, 10]
            colors = ["gray", "gray", "gray"]
            colors[self.winner_index] = "green"
            edge_colors = ["black", "black", "black"]
            edge_colors[self.final_choice] = "red"
            linewidths = [1, 1, 1]
            linewidths[self.final_choice] = 5
            door_numbers = ["Door 1", "Door 2", "Door 3"]
            self.ax.clear()
            for i in range(3):
                self.ax.text(self.prize_coordinates[i], 5, f"{self.prizes[i]}")
            self.ax.bar(
                door_numbers,
                values,
                color=colors,
                width=0.6,
                edgecolor=edge_colors,
                linewidth=linewidths,
            )
            self.game_over = True
            self.check_if_switch()
            msg = " " if self.switch else " NOT "
            self.ax.set_title(
                f"You decided{msg}to switch and chose door #{self.final_choice+1}\n You got a {self.prizes[self.final_choice]}"
            )

            self.games_finished += 1
            if self.switch:
                self.memory_wins["switch"] += self.doors[self.final_choice]
                self.memory_games["switch"] += 1
            else:
                self.memory_wins["stay"] += self.doors[self.final_choice]
                self.memory_games["stay"] += 1

            self.update_results_chart()

    def update_results_chart(self):
        self.results_ax.clear()
        self.results_ax.set_title(
            f"Games finished: {self.games_finished}\nGames you switched: {self.memory_games['switch']}, Games you stayed: {self.memory_games['stay']}"
        )
        self.results_ax.scatter(
            ["switch", "stay"],
            [
                self.memory_wins["switch"] / self.memory_games["switch"],
                self.memory_wins["stay"] / self.memory_games["stay"],
            ],
            s=350,
        )
        self.results_ax.set_ylim(0, 1)

    def check_if_switch(self):
        self.switch = False if self.choice == self.final_choice else True

    def init_monty_hall(self):
        doors = np.array([0, 0, 0])
        winner_index = np.random.randint(0, 3)
        doors[winner_index] = 1

        return doors, winner_index

    def open_door(self):
        openable_doors = [
            i for i in range(3) if i not in (self.winner_index, self.choice)
        ]
        door_to_open = np.random.choice(openable_doors)

        return door_to_open


def success_rate_plot(f):
    def _plot(switch, n_iterations):
        wins = 0
        # iterations = 1000

        for _ in range(n_iterations):
            wins += f(switch=switch)

        win_rate = wins / n_iterations
        loss_rate = 1 - win_rate

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.pie(
            [win_rate, loss_rate],
            labels=["Win a car", "Win... a goat?"],
            colors=sns.color_palette("pastel")[2:],
            autopct="%.0f%%",
        )

        msg = "always" if switch else "never"
        ax.set_title(f"Win rate if you {msg} switch doors ({n_iterations} simulations)")
        plt.show()

    def _plot_generalized(switch, n_iterations, n=3, k=1):
        wins = 0
        # iterations = 1000

        for _ in range(n_iterations):
            try:
                wins += f(switch=switch, n=n, k=k)
            except ValueError:
                print(
                    "n is the number of doors and k is the amount of doors the host opens. Since you have already picked one door, k has to be at most n-2, so there is at least one openable door after the host open the k doors."
                )
                return

        win_rate = wins / n_iterations
        loss_rate = 1 - win_rate

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.pie(
            [win_rate, loss_rate],
            labels=["Win a car", "Win... a goat?"],
            colors=sns.color_palette("pastel")[2:],
            autopct="%.0f%%",
        )

        msg = "always" if switch else "never"
        ax.set_title(f"Win rate if you {msg} switch doors ({n_iterations} simulations)")
        plt.show()

    n_iterations_selection = widgets.SelectionSlider(
        options=[1, 10, 100, 1000],
        value=1,
        description="# iterations",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
    )

    strategy_selection = widgets.RadioButtons(
        options=[True, False],
        value=False,
        description="Switch Doors?",
        disabled=False,
    )

    if f.__qualname__ == "monty_hall":
        interact_manual(
            _plot,
            switch=strategy_selection,
            n_iterations=n_iterations_selection,
        )

    if f.__qualname__ == "generalized_monty_hall":
        disabled = False

        n_selection = widgets.SelectionSlider(
            options=range(3, 101),
            value=3,
            description="n",
            disabled=disabled,
        )

        k_selection = widgets.SelectionSlider(
            options=range(0, 99),
            value=1,
            description="k",
            disabled=disabled,
        )

        interact_manual(
            _plot_generalized,
            switch=strategy_selection,
            n_iterations=n_iterations_selection,
            n=n_selection,
            k=k_selection,
        )


FEATURES = ["height", "weight", "bark_days", "ear_head_ratio"]


@dataclass
class params_gaussian:
    mu: float
    sigma: float

    def __repr__(self):
        return f"params_gaussian(mu={self.mu:.3f}, sigma={self.sigma:.3f})"


@dataclass
class params_binomial:
    n: int
    p: float

    def __repr__(self):
        return f"params_binomial(n={self.n:.3f}, p={self.p:.3f})"


@dataclass
class params_uniform:
    a: int
    b: int

    def __repr__(self):
        return f"params_uniform(a={self.a:.3f}, b={self.b:.3f})"


breed_params = {
    0: {
        "height": params_gaussian(mu=35, sigma=1.5),
        "weight": params_gaussian(mu=20, sigma=1),
        "bark_days": params_binomial(n=30, p=0.8),
        "ear_head_ratio": params_uniform(a=0.6, b=0.1),
    },
    1: {
        "height": params_gaussian(mu=30, sigma=2),
        "weight": params_gaussian(mu=25, sigma=5),
        "bark_days": params_binomial(n=30, p=0.5),
        "ear_head_ratio": params_uniform(a=0.2, b=0.5),
    },
    2: {
        "height": params_gaussian(mu=40, sigma=3.5),
        "weight": params_gaussian(mu=32, sigma=3),
        "bark_days": params_binomial(n=30, p=0.3),
        "ear_head_ratio": params_uniform(a=0.1, b=0.3),
    },
}


def round_dict(nested_dict):
    rounded_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            rounded_dict[key] = round_dict(value)
        else:
            rounded_dict[key] = round(value, 3)
    return rounded_dict


def generate_data_for_breed(breed, features, n_samples, params, gg, bg, ug):
    """
    Generate synthetic data for a specific breed of dogs based on given features and parameters.

    Parameters:
        - breed (str): The breed of the dog for which data is generated.
        - features (list[str]): List of features to generate data for (e.g., "height", "weight", "bark_days", "ear_head_ratio").
        - n_samples (int): Number of samples to generate for each feature.
        - params (dict): Dictionary containing parameters for each breed and its features.

    Returns:
        - df (pandas.DataFrame): A DataFrame containing the generated synthetic data.
            The DataFrame will have columns for each feature and an additional column for the breed.
    """

    df = pd.DataFrame()

    for feature in features:
        match feature:
            case "height" | "weight":
                df[feature] = gg(
                    params[breed][feature].mu, params[breed][feature].sigma, n_samples
                )

            case "bark_days":
                df[feature] = bg(
                    params[breed][feature].n, params[breed][feature].p, n_samples
                )

            case "ear_head_ratio":
                df[feature] = ug(
                    params[breed][feature].a, params[breed][feature].b, n_samples
                )

    df["breed"] = breed

    return df


def generate_data(gaussian_generator, binomial_generator, uniform_generator):
    # Generate data for each breed
    df_0 = generate_data_for_breed(
        breed=0,
        features=FEATURES,
        n_samples=1200,
        params=breed_params,
        gg=gaussian_generator,
        bg=binomial_generator,
        ug=uniform_generator,
    )
    df_1 = generate_data_for_breed(
        breed=1,
        features=FEATURES,
        n_samples=1350,
        params=breed_params,
        gg=gaussian_generator,
        bg=binomial_generator,
        ug=uniform_generator,
    )
    df_2 = generate_data_for_breed(
        breed=2,
        features=FEATURES,
        n_samples=900,
        params=breed_params,
        gg=gaussian_generator,
        bg=binomial_generator,
        ug=uniform_generator,
    )

    # Concatenate all breeds into a single dataframe
    df_all_breeds = pd.concat([df_0, df_1, df_2]).reset_index(drop=True)

    # Shuffle the data
    df_all_breeds = df_all_breeds.sample(frac=1)

    return df_all_breeds


def compute_training_params(df, features):
    """
    Computes the estimated parameters for training a model based on the provided dataframe and features.

    Args:
        df (pandas.DataFrame): The dataframe containing the training data.
        features (list): A list of feature names to consider.

    Returns:
        - params_dict (dict): A dictionary that contains the estimated parameters for each breed and feature.
    """

    # Dict that should contain the estimated parameters
    params_dict = {}

    ### START CODE HERE ###

    # Loop over the breeds
    for breed in range(3):  # @REPLACE for None in None:
        # Slice the original df to only include data for the current breed and the feature columns
        # For reference in slicing with pandas, you can use the df_breed.groupby function followed by .get_group
        # or you can use the syntax df[df['breed'] == group]
        df_breed = df[df["breed"] == breed][
            features
        ]  # @REPLACE df_breed = df[df["breed"] == None][features]

        # Initialize the inner dict
        inner_dict = {}

        # Loop over the columns of the sliced dataframe
        # You can get the columns of a dataframe like this: dataframe.columns
        for col in df_breed.columns:
            match col:
                case "height" | "weight":
                    mu, sigma = estimate_gaussian_params(df_breed[col])
                    m = {"mu": mu, "sigma": sigma}

                case "bark_days":
                    n, p = estimate_binomial_params(df_breed[col])
                    m = {"n": n, "p": p}

                case "ear_head_ratio":
                    a, b = estimate_uniform_params(df_breed[col])
                    m = {"a": a, "b": b}

            # Save the dataclass object within the inner dict
            inner_dict[col] = m

        # Save inner dict within outer dict
        params_dict[breed] = inner_dict

    ### END CODE HERE ###

    return params_dict


def estimate_gaussian_params(sample):
    ### START CODE HERE ###
    mu = np.mean(sample)
    sigma = np.std(sample)
    ### END CODE HERE ###

    return mu, sigma


def estimate_binomial_params(sample):
    ### START CODE HERE ###
    n = 30
    p = (sample / n).mean()
    ### END CODE HERE ###

    return n, p


def estimate_uniform_params(sample):
    ### START CODE HERE ###
    a = sample.min()
    b = sample.max()
    ### END CODE HERE ###

    return a, b


def plot_gaussian_distributions(gaussian_0, gaussian_1, gaussian_2):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(gaussian_0, alpha=0.5, label="gaussian_0", bins=32)
    ax.hist(gaussian_1, alpha=0.5, label="gaussian_1", bins=32)
    ax.hist(gaussian_2, alpha=0.5, label="gaussian_2", bins=32)
    ax.set_title("Histograms of Gaussian distributions")
    ax.set_xlabel("Values")
    ax.set_ylabel("Frequencies")
    ax.legend()
    plt.show()


def plot_binomial_distributions(binomial_0, binomial_1, binomial_2):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(binomial_0, alpha=0.5, label="binomial_0")
    ax.hist(binomial_1, alpha=0.5, label="binomial_1")
    ax.hist(binomial_2, alpha=0.5, label="binomial_2")
    ax.set_title("Histograms of Binomial distributions")
    ax.set_xlabel("Values")
    ax.set_ylabel("Frequencies")
    ax.legend()
    plt.show()


df_anscombe = pd.read_csv("df_anscombe.csv")
df_datasaurus = pd.read_csv("datasaurus.csv")


def plot_anscombes_quartet():
    fig, axs = plt.subplots(2, 2, figsize=(8, 5), tight_layout=True)
    i = 1
    fig.suptitle("Anscombe's quartet", fontsize=16)
    for line in axs:
        for ax in line:
            ax.scatter(
                df_anscombe[df_anscombe.group == i]["x"],
                df_anscombe[df_anscombe.group == i]["y"],
            )
            ax.set_title(f"Group {i}")
            ax.set_ylim(2, 15)
            ax.set_xlim(0, 21)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            i += 1


def display_widget():
    dropdown_graph_1 = widgets.Dropdown(
        options=df_datasaurus.group.unique(),
        value="dino",
        description="Data set 1: ",
        disabled=False,
    )

    statistics_graph_1 = widgets.Button(
        value=False,
        description="Compute stats",
        disabled=False,
        button_style="",
        tooltip="Description",
        icon="",
    )

    dropdown_graph_2 = widgets.Dropdown(
        options=df_datasaurus.group.unique(),
        value="h_lines",
        description="Data set 2: ",
        disabled=False,
    )

    statistics_graph_2 = widgets.Button(
        value=False,
        description="Compute stats",
        disabled=False,
        button_style="",
        tooltip="Description",
        icon="",
    )
    plotted_stats_graph_1 = None
    plotted_stats_graph_2 = None

    fig = plt.figure(figsize=(8, 4), tight_layout=True)
    gs = gridspec.GridSpec(2, 2)
    ax_1 = fig.add_subplot(gs[0, 0])
    ax_2 = fig.add_subplot(gs[1, 0])
    ax_text_1 = fig.add_subplot(gs[0, 1])
    ax_text_2 = fig.add_subplot(gs[1, 1])
    df_group_1 = df_datasaurus.groupby("group").get_group("dino")
    df_group_2 = df_datasaurus.groupby("group").get_group("h_lines")
    sc_1 = ax_1.scatter(df_group_1["x"], df_group_1["y"], s=4)
    sc_2 = ax_2.scatter(df_group_2["x"], df_group_2["y"], s=4)
    ax_1.set_xlabel("x")
    ax_1.set_ylabel("y")
    ax_2.set_xlabel("x")
    ax_2.set_ylabel("y")
    ax_text_1.axis("off")
    ax_text_2.axis("off")

    def dropdown_choice(value, plotted_stats, ax_text, sc):
        if value.new != plotted_stats:
            ax_text.clear()
            ax_text.axis("off")
        sc.set_offsets(df_datasaurus.groupby("group").get_group(value.new)[["x", "y"]])
        fig.canvas.draw_idle()

    def get_stats(value, plotted_stats, ax_text, dropdown, val):
        value = dropdown.value
        if value == plotted_stats:
            return
        ax_text.clear()
        ax_text.axis("off")
        df_group = df_datasaurus.groupby("group").get_group(value)
        means = df_group.mean()
        var = df_group.var()
        corr = df_group.corr()
        ax_text.text(
            0,
            0,
            f"Statistics:\n      Mean x:      {means['x']:.2f}\n      Variance x: {var['x']:.2f}\n\n      Mean y:      {means['y']:.2f}\n      Variance y: {var['y']:.2f}\n\n      Correlation:  {corr['x']['y']:.2f}",
        )
        if val == 1:
            plotted_stats_graph_1 = value
        if val == 2:
            plotted_stats_graph_2 = value

    dropdown_graph_1.observe(
        lambda value: dropdown_choice(value, plotted_stats_graph_1, ax_text_1, sc_1),
        names="value",
    )
    statistics_graph_1.on_click(
        lambda value: get_stats(
            value, plotted_stats_graph_1, ax_text_1, dropdown_graph_1, 1
        )
    )
    dropdown_graph_2.observe(
        lambda value: dropdown_choice(value, plotted_stats_graph_2, ax_text_2, sc_2),
        names="value",
    )
    statistics_graph_2.on_click(
        lambda value: get_stats(
            value, plotted_stats_graph_2, ax_text_2, dropdown_graph_2, 2
        )
    )
    graph_1_box = HBox([dropdown_graph_1, statistics_graph_1])
    graph_2_box = HBox([dropdown_graph_2, statistics_graph_2])
    display(VBox([graph_1_box, graph_2_box]))


def plot_datasaurus():
    fig, axs = plt.subplots(6, 2, figsize=(7, 9), tight_layout=True)
    i = 0
    fig.suptitle("Datasaurus", fontsize=16)
    for line in axs:
        for ax in line:
            if i > 12:
                ax.axis("off")
            else:
                group = df_datasaurus.group.unique()[i]
                ax.scatter(
                    df_datasaurus[df_datasaurus.group == group]["x"],
                    df_datasaurus[df_datasaurus.group == group]["y"],
                    s=4,
                )
                ax.set_title(f"Group {group}")
                ax.set_ylim(-5, 110)
                ax.set_xlim(10, 110)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                i += 1
