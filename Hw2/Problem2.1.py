import numpy as np

def kidney_shape(x, y):
    return (x**2 + y**2)**2 - (x**3 + y**3)

def is_inside_disc(x, y):
    return (x - 0.25)**2 + (y - 0.25)**2 <= 0.125

def rectangle_method(x_min, x_max, y_min, y_max, n):
    dx = (x_max - x_min) / n # Width of each rectangle
    dy = (y_max - y_min) / n # Height of each rectangle
    area = 0.0
    
    for i in range(n):
        for j in range(n):
            x = x_min + (i + 0.5) * dx  # Midpoint rule
            y = y_min + (j + 0.5) * dy
            if kidney_shape(x, y) <= 0 and not is_inside_disc(x, y):
                area += dx * dy
    
    return round(area, 4)

def trapezoidal_method(x_min, x_max, y_min, y_max, n):
    dx = (x_max - x_min) / n #Grid spacing in x-direction
    dy = (y_max - y_min) / n #Grid spacing in y-direction
    area = 0.0
    
    for i in range(n + 1):
        for j in range(n + 1):
            x = x_min + i * dx #Current x-coordinate
            y = y_min + j * dy #Current y-coordinate
            weight = 1 #Default weight (for inner points)
            if (i == 0 or i == n) and (j == 0 or j == n):
                weight = 0.25  # Corner points
            elif i == 0 or i == n or j == 0 or j == n:
                weight = 0.5  # Edge points
            
            if kidney_shape(x, y) <= 0 and not is_inside_disc(x, y):
                area += weight * dx * dy
    
    return round(area, 4)

def main():
  
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    n = 100  # Number of subdivisions

    area_rectangle = rectangle_method(x_min, x_max, y_min, y_max, n)
    area_trapezoidal = trapezoidal_method(x_min, x_max, y_min, y_max, n)
    print("n:", n)
    print("Remaining area using Rectangle Method:", area_rectangle)
    print("Remaining area using Trapezoidal Method:", area_trapezoidal)

if __name__ == "__main__":
    main()