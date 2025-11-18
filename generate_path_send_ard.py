#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import serial
import time
import matplotlib.pyplot as plt

# --- 1. CONFIGURE YOUR ROBOT ---
L1 = 82.0  # Length of arm 1 (e.g., in mm)
L2 = 100.0  # Length of arm 2 (e.g., in mm)
indent = 25  # Indent from edge of reachable area (in mm)
COUNTS_PER_ROTATION = 2100.0

# --- NEW SETTINGS ---
MOVE_TO_START_AT_BEGINNING = True
RETURN_TO_HOME_AT_END = True
Plotting_Freq = 0.002      # Time (in seconds) between points. 0.002 = 500Hz




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
        print(f"        Set cluster_steps = {cluster_steps}, linear_steps = {linear_steps}")
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
    diag_side = (2 * ((half_side)**2))**(1/2)
    c1 = (centre_x + half_side, centre_y + half_side) # top-left
    c2 = (centre_x - half_side, centre_y + half_side) # bottom-left
    c3 = (centre_x - half_side, centre_y - half_side) # bottom-right
    c4 = (centre_x + half_side, centre_y - half_side) # top-right
    # c1 = (centre_x + diag_side, centre_y) # top-left
    # c2 = (centre_x, centre_y + diag_side) # bottom-left
    # c3 = (centre_x - diag_side, centre_y) # bottom-right
    # c4 = (centre_x, centre_y - diag_side) # top-right
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
    v1 = (start_x, start_y)
    v2 = (start_x - height, start_y + (side_length/2))
    v3 = (start_x - height, start_y - (side_length/2))
    print(f"Triangle side: {steps_per_side} steps = {cluster_steps} (cluster) + {steps_per_side - 2*cluster_steps} (linear) + {cluster_steps} (cluster)")
    points = []
    points.extend(generate_eased_line(v1[0], v1[1], v2[0], v2[1], steps_per_side, cluster_steps)[:-1])
    points.extend(generate_eased_line(v2[0], v2[1], v3[0], v3[1], steps_per_side, cluster_steps)[:-1])
    points.extend(generate_eased_line(v3[0], v3[1], v1[0], v1[1], steps_per_side, cluster_steps))
    return points

# --- 5. NEW PLOTTING FUNCTION FOR REFERENCE VALUES ---
def plot_reference_counts(motor1_counts, motor2_counts, steps_start=0, steps_shape=0):
    """
    Plot the reference motor counts (positions) being sent to Arduino.
    
    Args:
        motor1_counts: List of motor 1 position counts
        motor2_counts: List of motor 2 position counts
        steps_start: Number of steps in the "move to start" phase
        steps_shape: Number of steps in the "shape" phase
    """
    
    N = len(motor1_counts)
    if N == 0:
        print("No motor counts to plot.")
        return
    
    # Create step indices for x-axis
    step_indices = list(range(N))
    
    # Create color coding based on phase
    colours = []
    for i in range(N):
        if i < steps_start:
            # Move to start phase - Blue
            colours.append('blue')
        elif i < steps_start + steps_shape:
            # Shape drawing phase - Green
            colours.append('green')
        else:
            # Return to home phase - Red
            colours.append('red')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Reference Motor Positions Sent to Arduino', fontsize=16)
    
    # Plot Motor 1 Reference
    ax1.scatter(step_indices, motor1_counts, c=colours, s=10, alpha=0.6)
    ax1.plot(step_indices, motor1_counts, 'k-', alpha=0.3, linewidth=0.5)
    ax1.set_ylabel('Motor 1 Position (Counts)')
    ax1.set_title('Motor 1 Reference Trajectory')
    ax1.grid(True, alpha=0.3)
    
    # Add phase markers for Motor 1
    if steps_start > 0:
        ax1.axvline(x=steps_start, color='blue', linestyle='--', alpha=0.5, label='Start of Shape')
    if steps_shape > 0:
        ax1.axvline(x=steps_start + steps_shape, color='red', linestyle='--', alpha=0.5, label='Start of Return')
    ax1.legend()
    
    # Plot Motor 2 Reference
    ax2.scatter(step_indices, motor2_counts, c=colours, s=10, alpha=0.6)
    ax2.plot(step_indices, motor2_counts, 'k-', alpha=0.3, linewidth=0.5)
    ax2.set_xlabel('Step Index')
    ax2.set_ylabel('Motor 2 Position (Counts)')
    ax2.set_title('Motor 2 Reference Trajectory')
    ax2.grid(True, alpha=0.3)
    
    # Add phase markers for Motor 2
    if steps_start > 0:
        ax2.axvline(x=steps_start, color='blue', linestyle='--', alpha=0.5, label='Start of Shape')
    if steps_shape > 0:
        ax2.axvline(x=steps_start + steps_shape, color='red', linestyle='--', alpha=0.5, label='Start of Return')
    ax2.legend()
    
    # Add text annotations for phases
    if steps_start > 0:
        ax1.text(steps_start/2, ax1.get_ylim()[1]*0.95, 'Move to Start', 
                 ha='center', va='top', fontsize=10, color='blue', alpha=0.7)
    if steps_shape > 0:
        ax1.text(steps_start + steps_shape/2, ax1.get_ylim()[1]*0.95, 'Draw Shape', 
                 ha='center', va='top', fontsize=10, color='green', alpha=0.7)
    if N > steps_start + steps_shape:
        ax1.text(steps_start + steps_shape + (N - steps_start - steps_shape)/2, 
                 ax1.get_ylim()[1]*0.95, 'Return Home', 
                 ha='center', va='top', fontsize=10, color='red', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":

    # --- CHOOSE YOUR SHAPE TO DRAW ---
    shape_to_plot = "S"   # "S" = Square, "C" = Circle, "T" = Triangle
    Plot_shape = False     # Whether to plot the shape (Set to True to verify layout)    
    Plot_perf = False      # Whether to plot the path (Set to True to verify layout)
    user = "Conor"
    #user = "Jamie"
    #user ="Hugo"
    
    # --- 1. NEW: SET PARAMETERS BASED ON SHAPE ---    
    if shape_to_plot == "S":
        Time_For_Drawing = 3.0      # Total time (in seconds) for *drawing the shape*
        cluster_ratio = 0.9         # Ratio of cluster steps to total steps for shapes
        cluster_ratio_start = 0.92   # Ratio of cluster steps to total steps for "move to start"
        #percentage_start = 20       # Percentage of *shape steps* for "move to start"
        #percentage_home = 20        # Percentage of *shape steps* for "return to home"
        STEPS_FOR_START_PATH = indent * 25
        STEPS_FOR_HOME_PATH = indent * 25        
    elif shape_to_plot == "C":
        Time_For_Drawing = 1.0
        cluster_ratio = 0.78         # Not used for circle
        cluster_ratio_start = 0.92   # Use 0.2 for a gentler start/end on the start path
        #percentage_start = 20
        #percentage_home = 20
        STEPS_FOR_START_PATH = indent * 15
        STEPS_FOR_HOME_PATH = indent * 15 

    elif shape_to_plot == "T":
        Time_For_Drawing = 2.0
        cluster_ratio = 0.9
        cluster_ratio_start = 0.78
        #percentage_start = 20
        #percentage_home = 20
        STEPS_FOR_START_PATH = indent * 15
        STEPS_FOR_HOME_PATH = indent * 15
    
    # --- 2. NEW: STEP CALCULATION (based on per-shape params) ---
    STEPS_FOR_SHAPE = int(Time_For_Drawing / Plotting_Freq)
    #STEPS_FOR_START_PATH = int(STEPS_FOR_SHAPE * (percentage_start / 100.0))
    #STEPS_FOR_START_PATH = 250
    #STEPS_FOR_HOME_PATH = int(STEPS_FOR_SHAPE * (percentage_home / 100.0))
    #STEPS_FOR_HOME_PATH = 250
    TOTAL_STEPS_SENT = STEPS_FOR_START_PATH + STEPS_FOR_SHAPE + STEPS_FOR_HOME_PATH

    print(f"--- Step Calculation ---")
    print(f"Shape Drawing Time: {Time_For_Drawing}s")
    print(f"Shape Steps: {STEPS_FOR_SHAPE} (at {Plotting_Freq*1000:.1f} ms/step)")
    #print(f"Start Path Steps: {STEPS_FOR_START_PATH} ({percentage_start} % of shape steps)")
    #print(f"Home Path Steps: {STEPS_FOR_HOME_PATH} ({percentage_home} % of shape steps)")
    print(f"TOTAL STEPS TO SEND: {TOTAL_STEPS_SENT}")
    print("--------------------------\n")
    
    
    xy_points = []
    path_name = "path_unknown"
    
    # --- 3. GENERATE SHAPE (Using STEPS_FOR_SHAPE) ---
    if shape_to_plot == "S":
        path_name = "path_square"
        # ... (rest of square logic unchanged)
        total_linear_steps = int(STEPS_FOR_SHAPE // (1/cluster_ratio))
        total_cluster_steps = STEPS_FOR_SHAPE - total_linear_steps
        linear_steps_per_side = total_linear_steps // 4
        cluster_steps_per_cluster = total_cluster_steps // 8
        steps_per_side_sq = linear_steps_per_side + (2 * cluster_steps_per_cluster)
        xy_points = generate_square(
            centre_x=L1 + L2 - (2*(((83)/2)**2))**(1/2) - indent, centre_y=0, side_length=83, 
            steps_per_side=steps_per_side_sq, 
            cluster_steps=cluster_steps_per_cluster
        )
    
    elif shape_to_plot == "C":
        path_name = "path_circle"
        xy_points = generate_circle(
            centre_x=L1 + L2 - 41 - indent, centre_y=0, radius=41, 
            steps=STEPS_FOR_SHAPE
        )

    elif shape_to_plot == "T":
        path_name = "path_triangle"
        # ... (rest of triangle logic unchanged)
        total_linear_steps = int(STEPS_FOR_SHAPE // (1/cluster_ratio))
        total_cluster_steps = STEPS_FOR_SHAPE - total_linear_steps
        linear_steps_per_side = total_linear_steps // 3
        cluster_steps_per_cluster = total_cluster_steps // 6
        steps_per_side_tri = linear_steps_per_side + (2 * cluster_steps_per_cluster)
        xy_points = generate_triangle(
            start_x=L1 + L2 - indent, start_y=0, side_length=97, 
            steps_per_side=steps_per_side_tri,
            cluster_steps=cluster_steps_per_cluster
        )
    
    # --- Process the chosen SHAPE path ---
    if not xy_points:
        print(f"No shape selected or points generated. Set 'shape_to_plot' to 'S', 'C', or 'T'.")
        exit()

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
    
# --- NEW FUNCTION (add this with other path generator functions) ---
def generate_clustered_t_values(total_steps, cluster_steps):
    """
    Generate t values (0 to 1) with clustering at the start and end.
    
    Args:
        total_steps: Total number of steps/segments
        cluster_steps: Number of cluster points at each end
    
    Returns:
        List of t values from 0.0 to 1.0 with clustering
    """
    if total_steps <= 0:
        return [0.0]
    
    linear_steps = total_steps - (2 * cluster_steps)
    if linear_steps < 0:
        print(f"WARNING: Reducing cluster_steps for start path. {total_steps} total segments is not enough for {cluster_steps} cluster points per side.")
        cluster_steps = total_steps // 2
        linear_steps = total_steps - (2 * cluster_steps)
        print(f"        Set cluster_steps = {cluster_steps}, linear_steps = {linear_steps}")
    
    t_values = [0.0]
    
    # Add clustered points at the start
    for i in range(cluster_steps, 0, -1):
        t_values.append(0.5 ** i)
    
    t_start = 0.5 ** cluster_steps if cluster_steps > 0 else 0.0
    t_end = 1.0 - (0.5 ** cluster_steps) if cluster_steps > 0 else 1.0
    
    # Add linear middle section
    linear_segment_count = max(1, linear_steps)
    for i in range(1, linear_steps):
        fraction = i / linear_segment_count
        t_values.append(t_start + (t_end - t_start) * fraction)
    
    # Add clustered points at the end
    for i in range(1, cluster_steps + 1):
        t_values.append(1.0 - (0.5 ** i))
    
    t_values.append(1.0)
    
    # Sort to ensure monotonic progression
    t_values.sort()
    
    return t_values

# --- MODIFIED: Generate "Move to Start" WITH CLUSTERING ---
motor1_start_path = []
motor2_start_path = []

# --- DEFINE HOME POS ---
home_x = L1 + L2
home_y = 0.0
home_angles = calculate_ik(home_x, home_y, L1, L2)
c1_home = radians_to_counts(home_angles[0], COUNTS_PER_ROTATION)
c2_home = radians_to_counts(home_angles[1], COUNTS_PER_ROTATION) 

if MOVE_TO_START_AT_BEGINNING and motor1_counts:
    
    # Calculate cluster steps for start path
    # MODIFIED: cluster_ratio_start is now the ratio of *linear* steps
    total_linear_steps_start = int(STEPS_FOR_START_PATH * cluster_ratio_start)
    total_cluster_steps_start = STEPS_FOR_START_PATH - total_linear_steps_start
    cluster_steps_start = total_cluster_steps_start // 2  # Divide between start and end
    
    # Generate clustered t values
    t_values = generate_clustered_t_values(STEPS_FOR_START_PATH, cluster_steps_start)
    
    # 1. Get Target counts (first point of the shape)
    c1_start_shape = motor1_counts[0]
    c2_start_shape = motor2_counts[0]

    for t in t_values[:-1]:  # Exclude the last point (t=1.0) to avoid duplication
        c1 = c1_home + (c1_start_shape - c1_home) * t
        c2 = c2_home + (c2_start_shape - c2_home) * t
        
        motor1_start_path.append(int(round(c1))) 
        motor2_start_path.append(int(round(c2)))

    
    # --- MODIFIED: Generate "Return to Home" WITH CLUSTERING ---
    motor1_home_path = []
    motor2_home_path = []
    
    if RETURN_TO_HOME_AT_END and motor1_counts:

        # 1. Calculate cluster steps for home path (using same logic as start path)
        total_linear_steps_home = int(STEPS_FOR_HOME_PATH * cluster_ratio_start)
        total_cluster_steps_home = STEPS_FOR_HOME_PATH - total_linear_steps_home
        cluster_steps_home = total_cluster_steps_home // 2 # Divide between start and end

        # 2. Generate clustered t values
        t_values_home = generate_clustered_t_values(STEPS_FOR_HOME_PATH, cluster_steps_home)
        
        # 3. Get Target counts (last point of the shape)
        c1_end_shape = motor1_counts[-1]
        c2_end_shape = motor2_counts[-1]   

        # 4. Interpolate using t values
        # Skip the first point (t=0.0) to avoid duplicating the last point of the shape
        for t in t_values_home[1:]: 
            c1 = c1_end_shape + (c1_home - c1_end_shape) * t
            c2 = c2_end_shape + (c2_home - c2_end_shape) * t
            
            motor1_home_path.append(int(round(c1)))
            motor2_home_path.append(int(round(c2)))

        
        
    # --- NEW: Combine all paths ---
    final_motor1_counts = motor1_start_path + motor1_counts + motor1_home_path
    final_motor2_counts = motor2_start_path + motor2_counts + motor2_home_path
    
    # --- MODIFIED: Plot the reference motor counts being sent ---
    if Plot_shape:
        if final_motor1_counts:
            plot_reference_counts(
                final_motor1_counts, 
                final_motor2_counts,
                steps_start=len(motor1_start_path),
                steps_shape=len(motor1_counts)
            )
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
        time.sleep(0.5)
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
