
import math
import serial
import time
import matplotlib.pyplot as plt

# --- 1. CONFIGURE YOUR ROBOT ---
# (Must match your physical robot and Arduino code)

L1 = 145.0  # Length of arm 1 (e.g., in mm)
L2 = 100.0  # Length of arm 2 (e.g., in mm)
COUNTS_PER_ROTATION = 2100.0

# --- NEW SETTINGS ---
MOVE_TO_START_AT_BEGINNING = True
RETURN_TO_HOME_AT_END = True
Time_For_Drawing = 2.5  # Total time (in seconds) for the *entire* motion
Plotting_Freq = 0.002   # Time (in seconds) between points

# --- CHOOSE YOUR SHAPE TO DRAW ---
shape_to_plot = "T"  # "S" = Square, "C" = Circle, "T" = Triangle
Plot = False            # Whether to plot the path
square = False        # Set to False
circle = False         # Set to False
triangle = False      # Set to False
if shape_to_plot == "S":
    square = True
elif shape_to_plot == "C":
    circle = True
elif shape_to_plot == "T":
    triangle = True

# --- This logic is clearer ---
# We calculate total steps from the *total* time.
TOTAL_STEPS = int(Time_For_Drawing / Plotting_Freq)

# Allocate percentages of the total steps
STEPS_FOR_START_PATH = int(TOTAL_STEPS * 0.25)  # 10% for start path
STEPS_FOR_HOME_PATH = int(TOTAL_STEPS * 0.10)   # 10% for home path
REMAINING_STEPS = TOTAL_STEPS - STEPS_FOR_START_PATH - STEPS_FOR_HOME_PATH # 80% for shape
print(f"Total Steps: {TOTAL_STEPS}")
print(f"  > Start Path: {STEPS_FOR_START_PATH}")
print(f"  > Home Path:  {STEPS_FOR_HOME_PATH}")
print(f"  > Shape Path: {REMAINING_STEPS}")
# --- 2. CONFIGURE YOUR PATH ---


# Number of points to use for clustering at each corner
# CLUSTER_STEPS_PER_CORNER = int(TOTAL_STEPS / 6)  # e.g., 20 points approaching, 20 leaving

# --- 3. CORE FUNCTIONS (Kinematics) ---
# (Unchanged)

def calculate_ik(x, y, L1, L2):
    """
    Calculates the 'elbow-up' inverse kinematics for a 2-link arm.
    Returns (theta1_rad, theta2_rad) or None if unreachable.
    """
    try:
        D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
        
        if not (-1 <= D <= 1):
            print(f"WARNING: Point (x={x}, y={y}) is unreachable.")
            return None

        theta2_rad = math.atan2(-math.sqrt(1 - D**2), D)
        theta1_rad = math.atan2(y, x) - math.atan2(L2 * math.sin(theta2_rad), L1 + L2 * math.cos(theta2_rad))
        
        return (theta1_rad, theta2_rad)
        
    except ValueError as e:
        print(f"Error calculating IK for (x={x}, y={y}): {e}")
        return None

def radians_to_counts(rad, counts_per_rotation):
    """Converts an angle in radians to encoder counts."""
    rotations = rad / (2.0 * math.pi)
    return int(rotations * counts_per_rotation)

def format_as_c_array(name, data_list):
    """Prints a list of numbers as a C++ array for Arduino."""
    print(f"// Path data for {name}")
    print(f"const int {name}[] PROGMEM = {{")
    
    for i in range(0, len(data_list), 10):
        line = ", ".join(map(str, data_list[i:i+10]))
        print(f"    {line},")
    print("};")
    print(f"const int {name}_count = {len(data_list)};")
    print("\n")

# --- 4. PATH GENERATOR FUNCTIONS ---

def generate_line(x_start, y_start, x_end, y_end, steps):
    """Generates a list of (x, y) points for a straight line."""
    points = []
    if steps == 0:
        return [(x_start, y_start)]
        
    for i in range(steps + 1):
        fraction = i / steps
        x = x_start + (x_end - x_start) * fraction
        y = y_start + (y_end - y_start) * fraction
        points.append((x, y))
    return points

def generate_eased_line(x_start, y_start, x_end, y_end, steps, cluster_steps):
    """
    Generates 'steps + 1' points, clustered at start and end
    based on the "half-distance" midpoint algorithm.
    """
    points = []
    (x_s, y_s) = (x_start, y_start)
    (x_e, y_e) = (x_end, y_end)
    
    if steps <= 0:
        return [(x_s, y_s)]

    linear_steps = steps - (2 * cluster_steps)
    
    if linear_steps < 0:
        print(f"WARNING: Reducing cluster_steps. {steps} total segments is not enough for {cluster_steps} cluster points per side.")
        cluster_steps = steps // 2
        linear_steps = steps - (2 * cluster_steps)
        print(f"         Set cluster_steps = {cluster_steps}, linear_steps = {linear_steps}")
        
    t_values = [0.0]  # t=0 (Point 1: Start)

    # 'Leaving' cluster
    for i in range(cluster_steps, 0, -1):
        t_values.append(0.5 ** i)

    # 'Linear' section
    t_start = 0.5 ** cluster_steps if cluster_steps > 0 else 0.0
    t_end = 1.0 - (0.5 ** cluster_steps) if cluster_steps > 0 else 1.0
    
    for i in range(1, linear_steps):
        fraction = i / linear_steps
        t_values.append(t_start + (t_end - t_start) * fraction)

    # 'Approaching' cluster
    for i in range(1, cluster_steps + 1):
        t_values.append(1.0 - (0.5 ** i))

    t_values.append(1.0)  # t=1 (Point N: End)
    
    # Generate points from the 't' values
    for t in t_values:
        x = x_s + (x_e - x_s) * t
        y = y_s + (y_e - y_s) * t
        points.append((x, y))
        
    return points

def generate_square(centre_x, centre_y, side_length, steps_per_side, cluster_steps):
    """Generates (x, y) points for a square with clustered corner points."""
    half_side = side_length / 2.0
    c1 = (centre_x - half_side, centre_y - half_side)
    c2 = (centre_x + half_side, centre_y - half_side)
    c3 = (centre_x + half_side, centre_y + half_side)
    c4 = (centre_x - half_side, centre_y + half_side)
    
    print(f"Square side: {steps_per_side} steps = {cluster_steps} (cluster) + {steps_per_side - 2*cluster_steps} (linear) + {cluster_steps} (cluster)")
    
    points = []
    points.extend(generate_eased_line(c1[0], c1[1], c2[0], c2[1], steps_per_side, cluster_steps)[:-1])
    points.extend(generate_eased_line(c2[0], c2[1], c3[0], c3[1], steps_per_side, cluster_steps)[:-1])
    points.extend(generate_eased_line(c3[0], c3[1], c4[0], c4[1], steps_per_side, cluster_steps)[:-1])
    points.extend(generate_eased_line(c4[0], c4[1], c1[0], c1[1], steps_per_side, cluster_steps))
    return points

def generate_circle(centre_x, centre_y, radius, steps):
    """Generates (x, y) points for a circle."""
    points = []
    for i in range(steps + 1):
        angle_rad = (i / steps) * 2.0 * math.pi
        x = centre_x + radius * math.cos(angle_rad)
        y = centre_y + radius * math.sin(angle_rad)
        points.append((x, y))
    return points

def generate_triangle(start_x, start_y, side_length, steps_per_side, cluster_steps):
    """Generates (x, y) points for an equilateral triangle with clustered corner points."""
    height = side_length * (math.sqrt(3) / 2.0)
    v1 = (start_x, start_y)
    v2 = (start_x + side_length, start_y)
    v3 = (start_x + side_length / 2.0, start_y + height)
    
    print(f"Triangle side: {steps_per_side} steps = {cluster_steps} (cluster) + {steps_per_side - 2*cluster_steps} (linear) + {cluster_steps} (cluster)")
    
    points = []
    points.extend(generate_eased_line(v1[0], v1[1], v2[0], v2[1], steps_per_side, cluster_steps)[:-1])
    points.extend(generate_eased_line(v2[0], v2[1], v3[0], v3[1], steps_per_side, cluster_steps)[:-1])
    points.extend(generate_eased_line(v3[0], v3[1], v1[0], v1[1], steps_per_side, cluster_steps))
    return points

# --- 5. PLOTTING FUNCTION ---
# (Unchanged)

def plot_path_with_colours(xy_points, arm_lengths=(L1, L2)):
    """
    Generates a plot of the (x,y) path, colour-coded by order.
    Goes from Red -> Green -> Blue.
    """
    print("Plotting generated path...")
    
    x_vals = [p[0] for p in xy_points]
    y_vals = [p[1] for p in xy_points]
    
    N = len(xy_points)
    if N <= 1:
        print("Not enough points to plot.")
        return
        
    colours = []
    for i in range(N):
        fraction = i / (N - 1)
        if fraction < 0.5:
            r = 1.0 - 2.0 * fraction
            g = 2.0 * fraction
            b = 0.0
        else:
            r = 0.0
            g = 1.0 - 2.0 * (fraction - 0.5)
            b = 2.0 * (fraction - 0.5)
        colours.append((r, g, b))
        
    plt.figure(figsize=(10, 8))
    plt.scatter(x_vals, y_vals, c=colours, s=10)
    
    # plt.plot(x_vals[0], y_vals[0], 'o', c='red', markersize=10, label='Start (Red)', mfc='none', mew=2)
    # plt.plot(x_vals[N//2], y_vals[N//2], 'o', c='green', markersize=10, label='Mid (Green)', mfc='none', mew=2)
    # plt.plot(x_vals[-1], y_vals[-1], 'o', c='blue', markersize=10, label='End (Blue)', mfc='none', mew=2)
    
    if arm_lengths:
        l1, l2 = arm_lengths
        max_reach = l1 + l2
        min_reach = abs(l1 - l2)
        circle_max = plt.Circle((0, 0), max_reach, color='gray', fill=False, linestyle='--', label=f'Max Reach ({max_reach}mm)')
        circle_min = plt.Circle((0, 0), min_reach, color='gray', fill=False, linestyle=':', label=f'Min Reach ({min_reach}mm)')
        plt.gca().add_artist(circle_max)
        if min_reach > 0:
            plt.gca().add_artist(circle_min)
        
    plt.title('Generated Path (Colour-coded by order)')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    print("Showing plot. Close the plot window to continue and send data to robot...")
    plt.show()


# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    
    # --- CHOOSE YOUR SHAPE ---
    
    # 1. Square
    if square:
        print("Generating Square Path...")
        path_name = "path_square"
        # --- 50/50 logic for square (4 sides, 8 clusters) ---
        total_linear_steps = REMAINING_STEPS // 2
        total_cluster_steps = REMAINING_STEPS - total_linear_steps
        linear_steps_per_side = total_linear_steps // 4
        cluster_steps_per_cluster = total_cluster_steps // 8
        steps_per_side_sq = linear_steps_per_side + (2 * cluster_steps_per_cluster)
        xy_points = generate_square(
            centre_x=120, centre_y=0, side_length=83, 
            steps_per_side=steps_per_side_sq, 
            cluster_steps=cluster_steps_per_cluster
        )
    
    # 2. Circle (Does not use easing)
    if circle:
        print("Generating Circle Path...")
        path_name = "path_circle"
        # Circle just uses all remaining steps
        xy_points = generate_circle(centre_x=180, centre_y=0, radius=41, steps=REMAINING_STEPS)

    # 3. Equilateral Triangle
    if triangle:
        print("Generating Triangle Path...")
        path_name = "path_triangle"
        # --- CORRECTED 50/50 logic for triangle (3 sides, 6 clusters) ---
        total_linear_steps = REMAINING_STEPS // 2
        total_cluster_steps = REMAINING_STEPS - total_linear_steps
        
        linear_steps_per_side = total_linear_steps // 3
        cluster_steps_per_cluster = total_cluster_steps // 6
        
        # This is the total number of steps to generate for one side
        steps_per_side_tri = linear_steps_per_side + (2 * cluster_steps_per_cluster)
        
        xy_points = generate_triangle(
            start_x=100, start_y=50, side_length=97, 
            steps_per_side=steps_per_side_tri,
            cluster_steps=cluster_steps_per_cluster # Pass the steps for *one* cluster
        )

    
    # --- *** NEW: Add "Move to Start" path if enabled *** ---
    if MOVE_TO_START_AT_BEGINNING and xy_points:
        print(f"Adding 'Move to Start' sequence ({STEPS_FOR_START_PATH} steps)...")
        
        # Get the first (x, y) point of the shape
        start_x, start_y = xy_points[0]
        
        # Define the (x, y) of the "Home" position (arm fully extended)
        home_x = L1 + L2
        home_y = 0.0

        # Generate a line from home to the start of the shape
        start_path_points = generate_line(home_x, home_y, start_x, start_y, STEPS_FOR_START_PATH)
        
        # Prepend these points. Drop the last point [:-1] of the start path
        # because it's identical to the first point of the main shape path.
        xy_points = start_path_points[:-1] + xy_points
        

    # --- Add "Return to Home" path if enabled ---
    if RETURN_TO_HOME_AT_END and xy_points:
        print(f"Adding 'Return to Home' sequence ({STEPS_FOR_HOME_PATH} steps)...")
        
        last_x, last_y = xy_points[-1]
        
        home_x = L1 + L2
        home_y = 0.0

        # Use the original 'generate_line' for a simple, linear return
        return_path_points = generate_line(last_x, last_y, home_x, home_y, STEPS_FOR_HOME_PATH)[1:]
        
        xy_points.extend(return_path_points)

    
    # --- Plot the generated path ---
    if Plot:
        if xy_points:
            plot_path_with_colours(xy_points, arm_lengths=(L1, L2))
        else:
            print("No (x, y) points were generated. Exiting.")
            exit()

    
    # --- Process the chosen path ---
    print("Calculating Inverse Kinematics...")
    motor1_counts = []
    motor2_counts = []
    
    for x, y in xy_points:
        angles = calculate_ik(x, y, L1, L2)
        
        if angles:
            theta1, theta2 = angles
            count1 = radians_to_counts(theta1, COUNTS_PER_ROTATION)
            count2 = radians_to_counts(theta2, COUNTS_PER_ROTATION)
            
            motor1_counts.append(count1)
            motor2_counts.append(-count2)
            
    # --- Print the C++ Arrays ---
    if motor1_counts:
        print("\n--- COPY AND PASTE THIS INTO YOUR ARDUINO SKETCH ---")
        print("#include <avr/pgmspace.h> // Include this at the top!\n")
        
        format_as_c_array(f"{path_name}_motor1", motor1_counts)
        format_as_c_array(f"{path_name}_motor2", motor2_counts)
        print("--- END OF ARDUINO DATA ---")
    else:
        print("No valid path data was generated.")
        exit()
        
        
# === CONFIGURE SERIAL PORT ===
try:
    ser = serial.Serial('COM5', 230400, timeout=1)
    time.sleep(2)
except serial.SerialException as e:
    print(f"\n--- ERROR: Could not open serial port ---")
    print(f"Details: {e}")
    print("Please check your port name and connection.")
    print("Path data was generated and plotted, but not sent.")
    exit()


# === Send the path ===
print("\nStreaming path to Pico...")

try:
    for a1, a2 in zip(motor1_counts, motor2_counts):
        line = f"{a1} {a2}\n"
        ser.write(line.encode('utf-8'))
        print(line.strip())
        response = ser.readline().decode('utf-8').strip()
        if response:
            print(f"  ^-- Pico response: {response}")
        time.sleep(Plotting_Freq)
except serial.SerialException as e:
    print(f"Serial error during streaming: {e}")
finally:
    print("Done streaming.")
    ser.close()
