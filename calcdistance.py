from sklearn.cluster import DBSCAN
import numpy as np

def analyze_vehicle_routes(locations, epsilon=50, min_samples=1):
    def count_vehicles(locations, epsilon, min_samples):
        # ... (same as previous count_vehicles function) ...
        return len(unique_labels), routes

    # Convert the list of (x, y) locations to a numpy array
    locations_np = np.array(locations)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(locations_np)

    # Get the unique labels (clusters) excluding outliers (label = -1)
    unique_labels = np.unique(labels[labels != -1])

    # Extract the routes for each vehicle
    routes = {}
    for label in unique_labels:
        vehicle_route = [tuple(locations_np[i]) for i in range(len(locations)) if labels[i] == label]
        routes[label] = vehicle_route

    return len(unique_labels), routes

def get_vehicle_routes(vehicle_routes):
    vehicle_routes_str = {}
    for vehicle_id, route in vehicle_routes.items():
        route_str = " -> ".join([f"({x},{y})" for x, y in route])
        vehicle_routes_str[vehicle_id] = route_str
    return vehicle_routes_str

# Example usage:
locations = [(495, 374), (281, 318), (751, 315), (780, 326), (736, 316), (535, 324), (497, 385), (991, 318),
             (514, 385), (777, 325), (514, 372), (776, 324), (282, 318), (737, 316), (581, 373), (983, 322),
             (768, 326), (658, 384), (753, 324)]


if __name__ == '__main__':
    num_vehicles, vehicle_routes = analyze_vehicle_routes(locations)
    print(f"Number of vehicles involved: {num_vehicles}")

    vehicle_routes_str = get_vehicle_routes(vehicle_routes)
    for vehicle_id, route_str in vehicle_routes_str.items():
        print(f"Vehicle {vehicle_id} route: {route_str}")