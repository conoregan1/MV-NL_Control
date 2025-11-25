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

def counts_to_radians(counts, counts_per_rotation):
    return (float(counts) / counts_per_rotation) * (2.0 * math.pi)

# --- 4. NEW GAUSSIAN PATH GENERATOR FUNCTIONS ---

def get_gaussian_t_values(steps, sigma):
    """
    Generates t values (0.0 to 1.0) based on the Error Function (erf).
    Result: Dense points at 0 and 1 (slow), sparse in middle (fast).
    """
    t_values = []
    if steps <= 0: return [0.0, 1.0]

    for i in range(steps + 1):
        # Map i to range -1.0 to 1.0
        raw_pos = -1.0 + (2.0 * i / steps)
        # Scale by sigma (standard deviation spread)
        x = raw_pos * sigma
        # Calculate erf (results in -1 to 1)
        val = math.erf(x)
        # Normalize to 0 to 1
        t = (val + 1.0) / 2.0
        t_values.append(t)
    
    # Force exact ends to avoid float drift
    t_values[0] = 0.0
    t_values[-1] = 1.0
    return t_values

def get_half_gaussian_t_values(steps, sigma, dense_at_end=True):
    """
    Generates a 'half curve' for entering/exiting the shape.
    dense_at_end=True:  Fast Start -> Slow/Dense End (Home -> Shape)
    dense_at_end=False: Slow/Dense Start -> Fast End (Shape -> Home)
    """
    t_values = []
    if steps <= 0: return [0.0, 1.0]
    
    # Normalization factor to ensure we end exactly at 1.0
    norm_factor = math.erf(sigma)

    for i in range(steps + 1):
        progress = i / steps # Linear 0.0 to 1.0
        
        if dense_at_end:
            # Input goes 0 -> sigma (Mean to Tail)
            x = progress * sigma
            t = math.erf(x) / norm_factor
        else:
            # Input goes sigma -> 0 (Tail to Mean)
            # We invert the logic of the line above
            x = (1.0 - progress) * sigma
            t = 1.0 - (math.erf(x) / norm_factor)
            
        t_values.append(t)

    t_values[0] = 0.0
    t_values[-1] = 1.0
    return t_values

def generate_gaussian_line(x_start, y_start, x_end, y_end, steps, sigma=2.5):
    points = []
    t_vals = get_gaussian_t_values(steps, sigma)
    
    for t in t_vals:
        x = x_start + (x_end - x_start) * t
        y = y_start + (y_end - y_start) * t
        points.append((x, y))
    return points

def generate_square(centre_x, centre_y, side_length, steps_per_side):
    half_side = side_length / 2.0
    c1 = (centre_x + half_side, centre_y + half_side) # top-left
    c2 = (centre_x - half_side, centre_y + half_side) # bottom-left
    c3 = (centre_x - half_side, centre_y - half_side) # bottom-right
    c4 = (centre_x + half_side, centre_y - half_side) # top-right
    
    points = []
    # Note: We slice [:-1] to avoid duplicating corner points
    points.extend(generate_gaussian_line(c1[0], c1[1], c2[0], c2[1], steps_per_side, GAUSSIAN_SIGMA)[:-1])
    points.extend(generate_gaussian_line(c2[0], c2[1], c3[0], c3[1], steps_per_side, GAUSSIAN_SIGMA)[:-1])
    points.extend(generate_gaussian_line(c3[0], c3[1], c4[0], c4[1], steps_per_side, GAUSSIAN_SIGMA)[:-1])
    points.extend(generate_gaussian_line(c4[0], c4[1], c1[0], c1[1], steps_per_side, GAUSSIAN_SIGMA))
    return points

def generate_circle(centre_x, centre_y, radius, steps):
    points = []
    # Circles are usually constant speed, but we can ensure steps are distributed
    for i in range(steps + 1):
        angle_rad = (i / steps) * 2.0 * math.pi
        x = centre_x + radius * math.cos(angle_rad)
        y = centre_y + radius * math.sin(angle_rad)
        points.append((x, y))
    return points

def generate_triangle(start_x, start_y, side_length, steps_per_side):
    height = side_length * (math.sqrt(3) / 2.0)
    v1 = (start_x, start_y)
    v2 = (start_x - height, start_y + (side_length/2))
    v3 = (start_x - height, start_y - (side_length/2))
    
    points = []
    points.extend(generate_gaussian_line(v1[0], v1[1], v2[0], v2[1], steps_per_side, GAUSSIAN_SIGMA)[:-1])
    points.extend(generate_gaussian_line(v2[0], v2[1], v3[0], v3[1], steps_per_side, GAUSSIAN_SIGMA)[:-1])
    points.extend(generate_gaussian_line(v3[0], v3[1], v1[0], v1[1], steps_per_side, GAUSSIAN_SIGMA))
    return points

# --- 5. PLOTTING FUNCTION ---
def plot_reference_counts(motor1_counts, motor2_counts, steps_start=0, steps_shape=0):
    N = len(motor1_counts)
    if N == 0: return
    
    step_indices = list(range(N))
    colours = []
    for i in range(N):
        if i < steps_start: colours.append('blue') # Move to Start
        elif i < steps_start + steps_shape: colours.append('green') # Shape
        else: colours.append('red') # Return Home
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Reference Motor Positions Sent to Arduino', fontsize=16)
    
    ax1.scatter(step_indices, motor1_counts, c=colours, s=10, alpha=0.6)
    ax1.plot(step_indices, motor1_counts, 'k-', alpha=0.3, linewidth=0.5)
    ax1.set_ylabel('Motor 1 Position')
    ax1.set_title('Motor 1 Reference')
    ax1.grid(True, alpha=0.3)
    
    if steps_start > 0: ax1.axvline(x=steps_start, color='blue', linestyle='--', alpha=0.5)
    if steps_shape > 0: ax1.axvline(x=steps_start + steps_shape, color='red', linestyle='--', alpha=0.5)

    ax2.scatter(step_indices, motor2_counts, c=colours, s=10, alpha=0.6)
    ax2.plot(step_indices, motor2_counts, 'k-', alpha=0.3, linewidth=0.5)
    ax2.set_ylabel('Motor 2 Position')
    ax2.set_title('Motor 2 Reference')
    ax2.grid(True, alpha=0.3)
    
    if steps_start > 0: ax2.axvline(x=steps_start, color='blue', linestyle='--', alpha=0.5)
    if steps_shape > 0: ax2.axvline(x=steps_start + steps_shape, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":

    # --- CHOOSE YOUR SHAPE TO DRAW ---
    shape_to_plot = "S"   # "S" = Square, "C" = Circle, "T" = Triangle
    Plot_shape = False     # Plot the planned path
    Plot_perf = False      # Plot the received performance data
    user = "Conor"  # "Conor", "Jamie", "Hugo" 

    # --- 1. PARAMETERS ---    
    if shape_to_plot == "S":
        Time_For_Drawing = 2.95
        STEPS_FOR_START_PATH = indent * 35
        STEPS_FOR_HOME_PATH = indent * 35   
        GAUSSIAN_SIGMA = 2.5      
    elif shape_to_plot == "C":
        Time_For_Drawing = 0.95
        STEPS_FOR_START_PATH = indent * 35
        STEPS_FOR_HOME_PATH = indent * 35 
        GAUSSIAN_SIGMA = 2.5
    elif shape_to_plot == "T":
        Time_For_Drawing = 1.95
        STEPS_FOR_START_PATH = indent * 35
        STEPS_FOR_HOME_PATH = indent * 35
        GAUSSIAN_SIGMA = 2.5
    
    STEPS_FOR_SHAPE = int(Time_For_Drawing / Plotting_Freq)
    TOTAL_STEPS_SENT = STEPS_FOR_START_PATH + STEPS_FOR_SHAPE + STEPS_FOR_HOME_PATH

    print(f"--- Step Calculation ---")
    print(f"Shape Steps: {STEPS_FOR_SHAPE}")
    print(f"Start/Home Steps: {STEPS_FOR_START_PATH} / {STEPS_FOR_HOME_PATH}")
    print("--------------------------\n")
    
    xy_points = []
    
    # --- 3. GENERATE SHAPE ---
    if shape_to_plot == "S":
        # Simple division by 4, logic handled inside function now
        xy_points = generate_square(
            centre_x=L1 + L2 - (2*(((83)/2)**2))**(1/2) - indent, centre_y=0, side_length=83, 
            steps_per_side=STEPS_FOR_SHAPE // 4
        )
    
    elif shape_to_plot == "C":
        xy_points = generate_circle(
            centre_x=L1 + L2 - 41 - indent, centre_y=0, radius=41, 
            steps=STEPS_FOR_SHAPE
        )

    elif shape_to_plot == "T":
        xy_points = generate_triangle(
            start_x=L1 + L2 - indent, start_y=0, side_length=97, 
            steps_per_side=STEPS_FOR_SHAPE // 3
        )
    
    if not xy_points:
        print(f"No shape generated.")
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
            motor2_counts.append(-count2) 
    
    
    # --- DEFINE HOME POS ---
    home_x = L1 + L2
    home_y = 0.0
    home_angles = calculate_ik(home_x, home_y, L1, L2)
    c1_home = radians_to_counts(home_angles[0], COUNTS_PER_ROTATION)
    c2_home = radians_to_counts(home_angles[1], COUNTS_PER_ROTATION) 

    # --- GENERATE PATHS WITH HALF-GAUSSIAN CLUSTERING ---
    motor1_start_path = []
    motor2_start_path = []
    motor1_home_path = []
    motor2_home_path = []

    if MOVE_TO_START_AT_BEGINNING and motor1_counts:
        # dense_at_end=True means: Sparse at Home, Dense at Shape Start
        t_values = get_half_gaussian_t_values(STEPS_FOR_START_PATH, GAUSSIAN_SIGMA, dense_at_end=True)
        
        c1_start_shape = motor1_counts[0]
        c2_start_shape = motor2_counts[0]

        for t in t_values[:-1]: # Exclude last point to avoid duplicate
            c1 = c1_home + (c1_start_shape - c1_home) * t
            c2 = c2_home + (c2_start_shape - c2_home) * t
            motor1_start_path.append(int(round(c1))) 
            motor2_start_path.append(int(round(c2)))

    if RETURN_TO_HOME_AT_END and motor1_counts:
        # dense_at_end=False means: Dense at Shape End, Sparse at Home
        t_values = get_half_gaussian_t_values(STEPS_FOR_HOME_PATH, GAUSSIAN_SIGMA, dense_at_end=False)
        
        c1_end_shape = motor1_counts[-1]
        c2_end_shape = motor2_counts[-1]   

        for t in t_values[1:]: # Skip first point to avoid duplicate
            c1 = c1_end_shape + (c1_home - c1_end_shape) * t
            c2 = c2_end_shape + (c2_home - c2_end_shape) * t
            motor1_home_path.append(int(round(c1)))
            motor2_home_path.append(int(round(c2)))

    # --- Combine ---
    final_motor1_counts = motor1_start_path + motor1_counts + motor1_home_path
    final_motor2_counts = motor2_start_path + motor2_counts + motor2_home_path
    
    if Plot_shape:
        plot_reference_counts(
            final_motor1_counts, final_motor2_counts,
            steps_start=len(motor1_start_path), steps_shape=len(motor1_counts)
        )
    
    # === SERIAL COMMUNICATION ===
    try:
        if user == "Conor":
            ser = serial.Serial('COM5', 230400, timeout=1)
        elif user == "Jamie":
            ser = serial.Serial('/dev/cu.usbmodem11401', 230400, timeout=1)
        elif user == "Hugo":
            ser = serial.Serial('/dev/cu.usbmodem1201', 230400, timeout=1)
        time.sleep(0.5)
    except serial.SerialException as e:
        print(f"Error opening serial: {e}")
        exit()

    # --- Send Data ---
    num_points = len(final_motor1_counts)
    print(f"\nSending {num_points} points to Pico...")
    ser.write(f"START {num_points}\n".encode('utf-8'))
    time.sleep(0.5)

    for a1, a2 in zip(final_motor1_counts, final_motor2_counts):
        ser.write(f"{a1} {a2}\n".encode('utf-8'))

    ser.write(b"END\n")
    print("Data sent. Reading log...\n")
    
    # --- Read Log ---
    try:
        end = False
        log_count = 0
        while not end:
            log_count += 1
            line_in = ser.readline().decode('utf-8').strip()
            
            if line_in:
                if line_in == "--- END LOG ---":
                    print(f"\n[Pico] {line_in}\n")
                    end = True
                    break
                if line_in == "--- BEGIN LOG ---":
                    print(f"\n[Pico] {line_in}\n")
                    continue

                try:
                    parts = line_in.split(',')
                    if len(parts) == 10:
                        step, ref1, e1, ref2, e2, uf_1, uf_2, u_p_2, u_d_2, u_i_2 = map(float, parts)
                        ref_1_data.append(ref1)
                        ref_2_data.append(ref2)
                        e_1_data.append(e1)
                        e_2_data.append(e2)
                        uf_prev_1_data.append(uf_1)
                        uf_prev_2_data.append(uf_2)
                        u_p_2_data.append(u_p_2)
                        u_d_2_data.append(u_d_2)
                        u_i_2_data.append(u_i_2)
                        print(f"Step: {int(step)} Motor 1: Ref: {ref1} Error: {e1} Effort: {uf_1} Motor 2: Ref: {ref2} Error: {e2} Effort: {uf_2} P: {u_p_2} I: {u_i_2} D: {u_d_2}")
                    else:
                        print(f"[Pico] {line_in}")
                except ValueError:
                    print(f"[Pico] Skip: {line_in}")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        ser.close()
        print(f"total log lines: {log_count}")
    
    # --- Plot Results ---
    if Plot_perf and ref_1_data:
        print("\nPlotting received data...")
        time_steps = range(len(ref_1_data))
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        ax1.plot(time_steps, ref_1_data, 'b.-', label='M1 Ref')
        ax1.plot(time_steps, ref_2_data, 'r.-', label='M2 Ref')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(time_steps, e_1_data, 'b--', label='M1 Err')
        ax2.plot(time_steps, e_2_data, 'r--', label='M2 Err')
        ax2.legend()
        ax2.grid(True)
        
        ax3.plot(time_steps, uf_prev_1_data, 'b.-', label='M1 Eff')
        ax3.plot(time_steps, uf_prev_2_data, 'r.-', label='M2 Eff')
        ax3.legend()
        ax3.grid(True)

        ax4.plot(time_steps, u_p_2_data, 'b.-', label='P')
        ax4.plot(time_steps, u_d_2_data, 'r.-', label='D')
        ax4.plot(time_steps, u_i_2_data, 'g.-', label='I')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
