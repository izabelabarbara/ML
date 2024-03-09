from flask import Flask, render_template, request, jsonify
import numpy as np
import random
import plotly.graph_objects as go
import plotly

# Ant Colony classes
def np_choice(a, size, p=None):
    return np.array(random.choices(a, weights=p, k=size))

class AntColony:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.distances = distances
        self.pheromone = np.ones(distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants, self.n_best, self.n_iterations = n_ants, n_best, n_iterations
        self.decay, self.alpha, self.beta = decay, alpha, beta

    def run(self):
        shortest_path = all_time_shortest_path = ("route", np.inf)
        for _ in range(self.n_iterations):
            all_paths = [self.gen_path() for _ in range(self.n_ants)]
            self.spread_pheromone(all_paths)
            shortest_path = min(all_paths, key=lambda x: x[1])
            all_time_shortest_path = min(all_time_shortest_path, shortest_path, key=lambda x: x[1])
            self.pheromone *= self.decay
        return all_time_shortest_path

    def spread_pheromone(self, all_paths):
        for path, dist in sorted(all_paths, key=lambda x: x[1])[:self.n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path(self):
        path, visited, prev = [], set(), 0
        visited.add(prev)
        for _ in range(len(self.distances) - 1):
            move = self.pick_move(prev, visited)
            path.append((prev, move))
            prev, visited = move, visited | {move}
        path.append((prev, 0))
        return path, self.path_dist(path)

    def pick_move(self, prev, visited):
        pheromone, dist = np.copy(self.pheromone[prev]), np.copy(self.distances[prev])
        pheromone[list(visited)], dist[dist == 0] = 0, 1
        prob = pheromone ** self.alpha * (1.0 / dist) ** self.beta
        if not np.isfinite(prob.sum()):
            print("Non-finite probabilities detected")
            print("Pheromone:", pheromone)
            print("Distance:", dist)
            print("Probabilities:", prob)
            prob = np.nan_to_num(prob)
        prob /= prob.sum()
        return np_choice(self.all_inds, 1, p=prob)[0]

    def path_dist(self, path):
        return sum(self.distances[move] for move in path)

class AntColonyVisualized(AntColony):
    def __init__(self, distances, coordinates, *args, **kwargs):
        super().__init__(distances, *args, **kwargs)
        self.coordinates = coordinates

    def plot(self, path):
        fig = go.Figure()
        for move in path:
            fig.add_trace(go.Scatter(x=[self.coordinates[i][0] for i in move],
                                     y=[self.coordinates[i][1] for i in move],
                                     mode='lines+markers',
                                     line=dict(color='blue')))
        fig.update_layout(title='Path found by Ant Colony',
                          xaxis_title='X Coordinate',
                          yaxis_title='Y Coordinate')
        return plotly.io.to_html(fig, full_html=False)

# Flask app setup
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        coordinates = np.array(data['points'])
        num_ants = int(data.get('num_ants', 10))
        decay_rate = float(data.get('decay_rate', 0.95))
        alpha = float(data.get('alpha', 1.0))
        beta = float(data.get('beta', 2.0))
        n_iterations = int(data.get('iterations', 100))
        initial_pheromone = float(data.get('initial_pheromone', 1.0))

        distances = np.sqrt(((coordinates[:, np.newaxis] - coordinates[np.newaxis, :]) ** 2).sum(axis=2))
        ant_colony = AntColonyVisualized(distances, coordinates, n_ants=num_ants, n_best=5,
                                         n_iterations=n_iterations, decay=decay_rate, alpha=alpha, beta=beta)
        ant_colony.pheromone = np.ones(distances.shape) * initial_pheromone

        shortest_path, length = ant_colony.run()
        plot_html = ant_colony.plot(shortest_path)
        path_coords = [coordinates[move[0]].tolist() for move in shortest_path]

        return jsonify({"plot_html": plot_html, "length": length, "shortest_path": path_coords})
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
