#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import serial
import time
import matplotlib.pyplot as plt

# --- 1. CONFIGURE YOUR ROBOT ---
L1 = 82.0  # Length of arm 1 (e.g., in mm)
L2 = 75.0  # Length of arm 2 (e.g., in mm)
COUNTS_PER_ROTATION = 2100.0

# --- NEW SETTINGS ---
MOVE_TO_START_AT_BEGINNING = True
RETURN_TO_HOME_AT_END = True
TOTAL_STEPS = 2000    # Total steps for the entire motion
Time_For_Drawing = 3  # Total time (in seconds) for the *entire* motion
Plotting_Freq = 0.002     # Time (in seconds) between points
cluster_ratio = 0.5   # Ratio of cluster steps to total steps for shapes, lower value = more cluster points
percentage_start = 20   # Percentage of total time for "move to start"
percentage_home = 10   # Percentage of total time for "return to home"

# --- NEW: CHOOSE YOUR INTERPOLATION METHOD ---
# "JOINT"     = Linear interpolation of motor counts (smoother for motors, curved in x,y)
# "CARTESIAN" = Linear interpolation of (x,y) coordinates (straight line in x,y, can be harsh for motors)
INTERPOLATION_MODE = "JOINT" 
#INTERPOLATION_MODE = "CARTESIAN"


# --- CHOOSE YOUR SHAPE TO DRAW ---
shape_to_plot = "C"   # "S" = Square, "C" = Circle, "T" = Triangle
Plot_shape = False    
Plot_perf = True      # Whether to plot the path (Set to True to verify layout)
user = "Conor"
#user = "Jamie"
#user = "Hugo"

#arrays for data storing
ref_1_data = []
ref_2_data = []
e_1_data = []
e_2_data = []
uf_prev_1_data = []
uf_prev_2_data = []
u_p_2_data = []
u_d_2_data = []
u_i_2_data = []

# --- Step Calculation ---
#TOTAL_STEPS = int(Time_For_Drawing / Plotting_Freq)
STEPS_FOR_START_PATH = int(TOTAL_STEPS * (percentage_start / 100.0))
STEPS_FOR_HOME_PATH = int(TOTAL_STEPS * (percentage_home / 100.0))
REMAINING_STEPS = TOTAL_STEPS - STEPS_FOR_START_PATH - STEPS_FOR_HOME_PATH
print(f"Total Steps: {TOTAL_STEPS}")
print(f"   > Start Path: {STEPS_FOR_START_PATH}")
print(f"   > Home Path:  {STEPS_FOR_HOME_PATH}")
print(f"   > Shape Path: {REMAINING_STEPS}")

# --- 3. CORE FUNCTIONS (Kinematics) ---
# (Unchanged)
def calculate_ik(x, y, L1, L2):
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
    rotations = rad / (2.0 * math.pi)
    return int(rotations * counts_per_rotation)

# (Unchanged)
def counts_to_radians(counts, counts_per_rotation):
    return (float(counts) / counts_per_rotation) * (2.0 * math.pi)

# (Unchanged)
def calculate_fk(theta1_rad, theta2_rad, L1, L2):
    x = L1 * math.cos(theta1_rad) + L2 * math.cos(theta1_rad + theta2_rad)
    y = L1 * math.sin(theta1_rad) + L2 * math.sin(theta1_rad + theta2_rad)
    return (x, y)

def format_as_c_array(name, data_list):
    print(f"// Path data for {name}")
    print(f"const int {name}[] PROGMEM = {{")
    for i in range(0, len(data_list), 10):
        line = ", ".join(map(str, data_list[i:i+10]))
        print(f"    {line},")
    print("};")
    print(f"const int {name}_count = {len(data_list)};")
    print("\n")

# --- 4. PATH GENERATOR FUNCTIONS ---
# (Unchanged)
def generate_line(x_start, y_start, x_end, y_end, steps):
    points = []
    if steps <= 0:
        return [(x_start, y_start)]
    for i in range(steps + 1):
        fraction = i / steps if steps > 0 else 0.0
        x = x_start + (x_end - x_start) * fraction
        y = y_start + (y_end - y_start) * fraction
        points.append((x, y))
    return points

def generate_eased_line(x_start, y_start, x_end, y_end, steps, cluster_steps):
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
    t_values = [0.0]
    for i in range(cluster_steps, 0, -1):
        t_values.append(0.5 ** i)
    t_start = 0.5 ** cluster_steps if cluster_steps > 0 else 0.0
    t_end = 1.0 - (0.5 ** cluster_steps) if cluster_steps > 0 else 1.0
    
    linear_segment_count = max(1, linear_steps)
    for i in range(1, linear_steps):
        fraction = i / linear_segment_count
        t_values.append(t_start + (t_end - t_start) * fraction)
    for i in range(1, cluster_steps + 1):
        t_values.append(1.0 - (0.5 ** i))
    t_values.append(1.0)
    
    # FIX: Sort the t_values to ensure monotonic progression
    t_values.sort()
    
    for t in t_values:
        x = x_s + (x_e - x_s) * t
        y = y_s + (y_e - y_s) * t
        points.append((x, y))
    return points

def generate_square(centre_x, centre_y, side_length, steps_per_side, cluster_steps):
    half_side = side_length / 2.0
    c1 = (centre_x + half_side, centre_y + half_side) 
    c2 = (centre_x - half_side, centre_y + half_side) 
    c3 = (centre_x - half_side, centre_y - half_side) 
    c4 = (centre_x + half_side, centre_y - half_side) 
    print(f"Square side: {steps_per_side} steps = {cluster_steps} (cluster) + {steps_per_side - 2*cluster_steps} (linear) + {cluster_steps} (cluster)")
    points = []
    points.extend(generate_eased_line(c1[0], c1[1], c2[0], c2[1], steps_per_side, cluster_steps)[:-1])
    points.extend(generate_eased_line(c2[0], c2[1], c3[0], c3[1], steps_per_side, cluster_steps)[:-1])
    points.extend(generate_eased_line(c3[0], c3[1], c4[0], c4[1], steps_per_side, cluster_steps)[:-1])
    points.extend(generate_eased_line(c4[0], c4[1], c1[0], c1[1], steps_per_side, cluster_steps))
    return points

def generate_circle(centre_x, centre_y, radius, steps):
    points = []
    for i in range(steps + 1):
        angle_rad = (i / steps) * 2.0 * math.pi
        x = centre_x + radius * math.cos(angle_rad)
        y = centre_y + radius * math.sin(angle_rad)
        points.append((x, y))
    return points

def generate_triangle(start_x, start_y, side_length, steps_per_side, cluster_steps):
    height = side_length * (math.sqrt(3) / 2.0)
    v3 = (start_x, start_y)
    v1 = (start_x + side_length, start_y)
    v2 = (start_x + side_length / 2.0, start_y + height)
    print(f"Triangle side: {steps_per_side} steps = {cluster_steps} (cluster) + {steps_per_side - 2*cluster_steps} (linear) + {cluster_steps} (cluster)")
    points = []
    points.extend(generate_eased_line(v1[0], v1[1], v2[0], v2[1], steps_per_side, cluster_steps)[:-1])
    points.extend(generate_eased_line(v2[0], v2[1], v3[0], v3[1], steps_per_side, cluster_steps)[:-1])
    points.extend(generate_eased_line(v3[0], v3[1], v1[0], v1[1], steps_per_side, cluster_steps))
    return points

# --- 5. PLOTTING FUNCTION ---
def plot_path_with_colours(xy_points, arm_lengths=(L1, L2)):
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
            r, g, b = 1.0 - 2.0 * fraction, 2.0 * fraction, 0.0
        else:
            r, g, b = 0.0, 1.0 - 2.0 * (fraction - 0.5), 2.0 * (fraction - 0.5)
        colours.append((r, g, b))
    plt.figure(figsize=(10, 8))
    plt.scatter(x_vals, y_vals, c=colours, s=10)
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
    print("Showing plot. Close the plot window to continue...")
    plt.show()
    



# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    
    xy_points = []
    path_name = "path_unknown"
    
    # --- CHOOSE YOUR SHAPE (This part is unchanged) ---
    if shape_to_plot == "S":
        print("Generating Square Path...")
        path_name = "path_square"
        # ... (rest of square logic unchanged)
        total_linear_steps = int(REMAINING_STEPS // (1/cluster_ratio))
        total_cluster_steps = REMAINING_STEPS - total_linear_steps
        linear_steps_per_side = total_linear_steps // 4
        cluster_steps_per_cluster = total_cluster_steps // 8
        steps_per_side_sq = linear_steps_per_side + (2 * cluster_steps_per_cluster)
        xy_points = generate_square(
            centre_x=100, centre_y=0, side_length=83, 
            steps_per_side=steps_per_side_sq, 
            cluster_steps=cluster_steps_per_cluster
        )
    
    elif shape_to_plot == "C":
        print("Generating Circle Path...")
        path_name = "path_circle"
        xy_points = generate_circle(
            centre_x=100, centre_y=0, radius=41, 
            steps=REMAINING_STEPS
        )

    elif shape_to_plot == "T":
        print("Generating Triangle Path...")
        path_name = "path_triangle"
        # ... (rest of triangle logic unchanged)
        total_linear_steps = int(REMAINING_STEPS // (1/cluster_ratio))
        total_cluster_steps = REMAINING_STEPS - total_linear_steps
        linear_steps_per_side = total_linear_steps // 3
        cluster_steps_per_cluster = total_cluster_steps // 6
        steps_per_side_tri = linear_steps_per_side + (2 * cluster_steps_per_cluster)
        xy_points = generate_triangle(
            start_x=50, start_y=-40, side_length=97, 
            steps_per_side=steps_per_side_tri,
            cluster_steps=cluster_steps_per_cluster
        )

    
    # --- (REMOVED) "Move to Start" and "Return to Home" from xy_points ---
    # We will do this in joint-space (counts) later.
    

    
    # --- Process the chosen SHAPE path ---
    if not xy_points:
        print(f"No shape selected or points generated. Set 'shape_to_plot' to 'S', 'C', or 'T'.")
        exit()

    print("Calculating Inverse Kinematics for SHAPE...")
    motor1_counts = []
    motor2_counts = []
    
    for x, y in xy_points:
        angles = calculate_ik(x, y, L1, L2)
        if angles:
            theta1, theta2 = angles
            count1 = radians_to_counts(theta1, COUNTS_PER_ROTATION)
            count2 = radians_to_counts(theta2, COUNTS_PER_ROTATION)
            motor1_counts.append(count1)
            motor2_counts.append(-count2) # Keeping your inversion
    
    
    # --- MODIFIED: Generate "Move to Start" ---
    motor1_start_path = []
    motor2_start_path = []
    
    # --- DEFINE HOME POS ---
    home_x = L1 + L2
    home_y = 0.0
    home_angles = calculate_ik(home_x, home_y, L1, L2)
    c1_home = radians_to_counts(home_angles[0], COUNTS_PER_ROTATION)
    c2_home = radians_to_counts(home_angles[1], COUNTS_PER_ROTATION) 
    
    if MOVE_TO_START_AT_BEGINNING and motor1_counts:
        print(f"Adding 'Move to Start' sequence ({STEPS_FOR_START_PATH} steps)...")
        print(f"   > Using {INTERPOLATION_MODE} interpolation.")
        
        # 1. Get Target counts (first point of the shape)
        c1_start_shape = motor1_counts[0]
        c2_start_shape = motor2_counts[0]

        if INTERPOLATION_MODE == "JOINT":
            # --- METHOD 1: JOINT SPACE INTERPOLATION ---
            for i in range(STEPS_FOR_START_PATH): # Creates points 0 to STEPS-1
                fraction = float(i) / STEPS_FOR_START_PATH
                
                c1 = c1_home + (c1_start_shape - c1_home) * fraction
                c2 = c2_home + (c2_start_shape - c2_home) * fraction
                
                motor1_start_path.append(int(round(c1))) 
                motor2_start_path.append(int(round(c2)))

        elif INTERPOLATION_MODE == "CARTESIAN":
            # --- METHOD 2: CARTESIAN SPACE INTERPOLATION ---
            (start_shape_x, start_shape_y) = xy_points[0]
            
            for i in range(STEPS_FOR_START_PATH): # Creates points 0 to STEPS-1
                fraction = float(i) / STEPS_FOR_START_PATH
                x = home_x + (start_shape_x - home_x) * fraction
                y = home_y + (start_shape_y - home_y) * fraction
                
                # Now run IK for this (x,y) point
                angles = calculate_ik(x, y, L1, L2)
                if angles:
                    theta1, theta2 = angles
                    count1 = radians_to_counts(theta1, COUNTS_PER_ROTATION)
                    count2 = radians_to_counts(theta2, COUNTS_PER_ROTATION)
                    motor1_start_path.append(count1)
                    motor2_start_path.append(-count2) # Keep inversion
                else:
                    # Handle unreachable point - append last valid point
                    print(f"Warning: Start path point {i} unreachable. Holding last position.")
                    if motor1_start_path:
                        motor1_start_path.append(motor1_start_path[-1])
                        motor2_start_path.append(motor2_start_path[-1])
                    else: # If first point is unreachable, use home counts
                        motor1_start_path.append(c1_home)
                        motor2_start_path.append(c2_home)
        else:
            print(f"ERROR: Unknown INTERPOLATION_MODE: '{INTERPOLATION_MODE}'")

    
    # --- MODIFIED: Generate "Return to Home" ---
    motor1_home_path = []
    motor2_home_path = []
    
    if RETURN_TO_HOME_AT_END and motor1_counts:
        print(f"Adding 'Return to Home' sequence ({STEPS_FOR_HOME_PATH} steps)...")
        print(f"   > Using {INTERPOLATION_MODE} interpolation.")
        
        # 1. Get Target counts (last point of the shape)
        c1_end_shape = motor1_counts[-1]
        c2_end_shape = motor2_counts[-1]
        
        if INTERPOLATION_MODE == "JOINT":
            # --- METHOD 1: JOINT SPACE INTERPOLATION ---
            for i in range(1, STEPS_FOR_HOME_PATH + 1): # Creates points 1 to STEPS
                fraction = float(i) / STEPS_FOR_HOME_PATH
                
                c1 = c1_end_shape + (c1_home - c1_end_shape) * fraction
                c2 = c2_end_shape + (c2_home - c2_end_shape) * fraction
                
                motor1_home_path.append(int(round(c1)))
                motor2_home_path.append(int(round(c2)))

        elif INTERPOLATION_MODE == "CARTESIAN":
            # --- METHOD 2: CARTESIAN SPACE INTERPOLATION ---
            (end_shape_x, end_shape_y) = xy_points[-1]
            
            for i in range(1, STEPS_FOR_HOME_PATH + 1): # Creates points 1 to STEPS
                fraction = float(i) / STEPS_FOR_HOME_PATH
                x = end_shape_x + (home_x - end_shape_x) * fraction
                y = end_shape_y + (home_y - end_shape_y) * fraction
                
                # Now run IK for this (x,y) point
                angles = calculate_ik(x, y, L1, L2)
                if angles:
                    theta1, theta2 = angles
                    count1 = radians_to_counts(theta1, COUNTS_PER_ROTATION)
                    count2 = radians_to_counts(theta2, COUNTS_PER_ROTATION)
                    motor1_home_path.append(count1)
                    motor2_home_path.append(-count2) # Keep inversion
                else:
                    # Handle unreachable point
                    print(f"Warning: Home path point {i} unreachable. Holding last position.")
                    if motor1_home_path:
                        motor1_home_path.append(motor1_home_path[-1])
                        motor2_home_path.append(motor2_home_path[-1])
                    else: # If first point is unreachable, use shape's end counts
                        motor1_home_path.append(c1_end_shape)
                        motor2_home_path.append(c2_end_shape)

            
            
    # --- NEW: Combine all paths ---
    final_motor1_counts = motor1_start_path + motor1_counts + motor1_home_path
    final_motor2_counts = motor2_start_path + motor2_counts + motor2_home_path

    # --- MODIFIED: Plot the FULL path using Forward Kinematics ---
    if Plot_shape:
        if final_motor1_counts:
            print("Calculating Forward Kinematics for full path plotting...")
            full_xy_path = []
            for c1, c2 in zip(final_motor1_counts, final_motor2_counts):
                # Convert counts back to the radians *used by the IK model*
                theta1_rad = counts_to_radians(c1, COUNTS_PER_ROTATION)
                # IMPORTANT: Invert c2 back to match the IK's theta2
                # (c2 is the *inverted* count, so -c2 is the original)
                theta2_rad = counts_to_radians(-c2, COUNTS_PER_ROTATION) 
                
                # Calculate the (x, y) position using FK
                x, y = calculate_fk(theta1_rad, theta2_rad, L1, L2)
                full_xy_path.append((x, y))
            
            # Now plot this new, complete list
            plot_path_with_colours(full_xy_path, arm_lengths=(L1, L2))
            
        else:
            print("No motor counts were generated. Cannot plot path.")
            exit()
    
    # === CONFIGURE SERIAL PORT (Unchanged) ===
    try:
        if user == "Conor":
            ser = serial.Serial('COM5', 230400, timeout=1)
        elif user == "Jamie":
            ser = serial.Serial('/dev/cu.usbmodem11401', 230400, timeout=1)
        elif user == "Hugo":
            ser = serial.Serial('/dev/cu.usbmodem1201', 230400, timeout=1)
        time.sleep(2)
    except serial.SerialException as e:
        print(f"\n--- ERROR: Could not open serial port ---")
        print(f"Details: {e}")
        exit()


    # === Send the full dataset (MODIFIED) ===
    num_points = len(final_motor1_counts) # Use the new final list
    print(f"\nSending {num_points} total points to Pico...")

    # --- Send header ---
    ser.write(f"START {num_points}\n".encode('utf-8'))
    time.sleep(0.5)

    # --- Send all positions (MODIFIED) ---
    for a1, a2 in zip(final_motor1_counts, final_motor2_counts): # Use the new final lists
        ser.write(f"{a1} {a2}\n".encode('utf-8'))

    ser.write(b"END\n")
    print("All data sent. Waiting for debug messages from Pico...\n")
    

# --- Read back debug info indefinitely ---
# --- THIS ENTIRE BLOCK IS MODIFIED ---
try:
    end = False
    log_count = 0
    while not end:
        log_count += 1
        line_in = ser.readline().decode('utf-8').strip()
        
        if line_in:
            # --- MODIFIED: Handle special, non-data lines first ---
            if line_in == "--- END LOG ---":
                print(f"\n[Pico] {line_in}\n") # Add newlines for clarity
                end = True
                break
            
            if line_in == "--- BEGIN LOG ---":
                print(f"\n[Pico] {line_in}\n") # Add newlines for clarity
                continue # Skip to the next line read

            # --- MODIFIED: Try to parse as data ---
            try:
                parts = line_in.split(',')
                if len(parts) == 10:
                    # 1. Parse the data
                    # We capture 'step' now instead of using '_'
                    step, ref1, e1, ref2, e2, uf_1, uf_2, u_p_2, u_d_2, u_i_2 = map(float, parts)
                    
                    # 2. Append data (same as before)
                    ref_1_data.append(ref1)
                    ref_2_data.append(ref2)
                    e_1_data.append(e1)
                    e_2_data.append(e2)
                    uf_prev_1_data.append(uf_1)
                    uf_prev_2_data.append(uf_2)
                    u_p_2_data.append(u_p_2)
                    u_d_2_data.append(u_d_2)
                    u_i_2_data.append(u_i_2)
                    
                    # 3. NEW: Print the labelled data
                    # Use int(step) for a cleaner print
                    # Use f-string formatting for aligned columns
                    print(f"Step: {int(step)} Motor 1: Ref: {ref1} Error: {e1} Effort: {uf_1} Motor 2: Ref: {ref2} Error: {e2} Effort: {uf_2} P: {u_p_2} I: {u_i_2} D: {u_d_2}")
                    
                else:
                    # Not a 10-part CSV line, print it normally
                    # (e.g., "Pico ready.", "RECV 500", "Path loaded.")
                    print(f"[Pico] {line_in}")
                    
            except ValueError:
                # Looked like data but failed to parse
                print(f"[Pico] Skipping unparseable line: {line_in}")

except KeyboardInterrupt:
    print("User interrupted, closing serial.")
finally:
    ser.close()
    # This print statement now counts all lines read,
    # including "--- BEGIN/END LOG ---" and other messages.
    print(f"{log_count} lines read from Pico.")


    
# === NEW: PLOT THE RECEIVED DATA ===
print("\nPlotting received data...")

if not ref_1_data:
    print("No data was received (or logged) from the Pico, cannot plot.")
else:
    # Create a time step array for the x-axis
    # Note: The x-axis will now be 'Logged Points', not 'Time Steps'
    time_steps = range(len(ref_1_data))

    # Create a figure with 3 subplots, sharing the x-axis
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Robot Arm Controller Performance (Logged 1 in 100 points)', fontsize=16)

    # --- Plot 1: Reference Positions ---
    ax1.plot(time_steps, ref_1_data, '.-', label='Motor 1 Reference', color='blue')
    ax1.plot(time_steps, ref_2_data, '.-', label='Motor 2 Reference', color='red')
    ax1.set_ylabel('Position (Counts)')
    ax1.set_title('Reference Positions (Target)')
    ax1.legend()
    ax1.grid(True)
    ref_1_data = []
    ref_2_data = []

    # --- Plot 2: Errors ---
    ax2.plot(time_steps, e_1_data, '.-', label='Motor 1 Error', color='blue', linestyle='--')
    ax2.plot(time_steps, e_2_data, '.-', label='Motor 2 Error', color='red', linestyle='--')
    ax2.set_ylabel('Error (Counts)')
    ax2.set_title('Following Error (Target - Actual)')
    ax2.legend()
    ax2.grid(True)
    e_1_data = []
    e_2_data = []
    
    # --- Plot 3: Control Effort ---
    ax3.plot(time_steps, uf_prev_1_data, '.-', label='Motor 1 Effort', color='blue')
    ax3.plot(time_steps, uf_prev_2_data, '.-', label='Motor 2 Effort', color='red')
    ax3.set_xlabel('Logged Point Index (1 per 100 steps)')
    ax3.set_ylabel('Control Signal')
    ax3.set_title('Control Effort (Output)')
    ax3.legend()
    ax3.grid(True)
    uf_prev_1_data = []
    uf_prev_2_data = []

    ax4.plot(time_steps, u_p_2_data, '.-', label='Proportional', color='blue')
    ax4.plot(time_steps, u_d_2_data, '.-', label='Derivative', color='red')
    ax4.plot(time_steps, u_i_2_data, '.-', label='Integral', color='green')
    ax4.set_xlabel('Logged Point Index (1 per 100 steps)')
    ax4.set_ylabel('Control Signal')
    ax4.set_title('Control Effort (Output)')
    ax4.legend()
    ax4.grid(True)
    u_p_2_data = []
    u_d_2_data = []
    u_i_2_data = []

    # Show the plot
    if Plot_perf:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
        plt.show()
