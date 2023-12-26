import numpy as np

def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def find_all_routes(locations, distance_threshold=100):
    def add_point_to_route(point, route):
        if all(euclidean_distance(point, p) <= distance_threshold for p in route):
            route.append(point)
            return True
        return False

    routes = []
    for p in locations:
        added_to_existing_route = False
        for route in routes:
            if add_point_to_route(p, route):
                added_to_existing_route = True
                break
        if not added_to_existing_route:
            routes.append([p])

    return routes

def count_routes(locations, distance_threshold=100):
    routes = find_all_routes(locations, distance_threshold)
    return len(routes)

# Example usage:
locations = [(495, 374), (281, 318), (751, 315), (780, 326), (736, 316), (535, 324), (497, 385), (991, 318),
             (514, 385), (777, 325), (514, 372), (776, 324), (282, 318), (737, 316), (581, 373), (983, 322),
             (768, 326), (658, 384), (753, 324)]

if __name__ == '__main__':
    num_routes = count_routes(locations, distance_threshold=100)
    print(f"Number of routes found: {num_routes}")

    routes = find_all_routes(locations, distance_threshold=100)
    for idx, route in enumerate(routes):
        route_str = " -> ".join([f"({x},{y})" for x, y in route])
        print(f"Route {idx + 1}: {route_str}")