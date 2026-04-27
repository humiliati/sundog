"""
Sundog Leisure Environment Generator

Creates a hexagonal lattice ceiling with overlapping spheres for maximum light 
convergence/refraction. The agent is recumbent (lying down) so stability is 
trivially cheap, freeing resources for tinkering with blocks that modulate 
the bloom field.

Key Design Principles:
1. Agent is horizontal - no balance problem
2. Hexagonal ceiling creates natural attractor points for shadow collapse  
3. Pushable blocks create indirect shadow feedback
4. Gravity becomes a toy, not a constraint
"""

import math
import numpy as np

def generate_hex_lattice(center_x, center_y, radius, spacing, z_base, z_amplitude=0.03):
    """
    Generate hexagonal lattice positions with overlapping spheres.
    The offset rows create natural light convergence points.
    
    Args:
        center_x, center_y: Center of the lattice region
        radius: Extent of the lattice from center
        spacing: Distance between sphere centers (should be < 2*sphere_radius for overlap)
        z_base: Base height of ceiling
        z_amplitude: Wave amplitude for 3D texture
    
    Returns:
        List of (x, y, z) positions
    """
    positions = []
    row_height = spacing * math.sqrt(3) / 2
    
    # Calculate bounds
    y_start = center_y - radius
    y_end = center_y + radius
    x_start = center_x - radius
    x_end = center_x + radius
    
    row = 0
    y = y_start
    while y <= y_end:
        # Offset every other row by half spacing (hexagonal pattern)
        x_offset = (spacing / 2) if row % 2 == 1 else 0
        x = x_start + x_offset
        
        while x <= x_end:
            # Distance from center for radial harmonic
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Create interference pattern: concentric + hex-aligned waves
            z_wave = z_amplitude * (
                math.sin(dist * 2 * math.pi / (spacing * 6)) *  # Radial wave
                math.cos(x * math.pi / spacing) *                 # X-aligned wave
                math.cos(y * math.pi / (row_height * 2))          # Y-aligned wave
            )
            
            z = z_base + z_wave
            positions.append((x, y, z))
            x += spacing
        
        y += row_height
        row += 1
    
    return positions


def generate_play_blocks(num_blocks=8, arena_radius=1.5, z_floor=0.05):
    """
    Generate pushable blocks for the play area.
    Various sizes for different shadow effects.
    """
    blocks = []
    np.random.seed(42)  # Reproducible
    
    block_types = [
        {'size': (0.08, 0.08, 0.08), 'mass': 0.1, 'color': '0.9 0.4 0.3 1'},   # Small red
        {'size': (0.12, 0.12, 0.12), 'mass': 0.2, 'color': '0.4 0.9 0.3 1'},   # Medium green
        {'size': (0.06, 0.06, 0.15), 'mass': 0.15, 'color': '0.3 0.4 0.9 1'},  # Tall blue
        {'size': (0.15, 0.15, 0.06), 'mass': 0.25, 'color': '0.9 0.9 0.3 1'},  # Flat yellow
    ]
    
    for i in range(num_blocks):
        angle = 2 * math.pi * i / num_blocks + np.random.uniform(-0.3, 0.3)
        r = arena_radius * np.random.uniform(0.3, 0.8)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        
        bt = block_types[i % len(block_types)]
        z = z_floor + bt['size'][2]
        
        blocks.append({
            'name': f'block_{i}',
            'pos': (x, y, z),
            'size': bt['size'],
            'mass': bt['mass'],
            'color': bt['color']
        })
    
    return blocks


def generate_recumbent_agent_xml():
    """
    Generate XML for a recumbent (lying) agent with tinkering limbs.
    
    The agent lies on its back, with:
    - Two arm-like appendages for pushing/manipulating
    - A "head" with sensor tip for bloom detection
    - Minimal DOF to keep proprioception lightweight
    """
    return '''
        <!-- Recumbent Agent: Lying on back, no balance needed -->
        <body name="torso" pos="0 0 0.08">
            <freejoint name="torso_free"/>
            <geom name="torso_geom" type="capsule" fromto="0 -0.15 0 0 0.15 0" size="0.06" 
                  rgba="0.5 0.5 0.6 1" mass="1.0"/>
            
            <!-- Head with sensor tip (bloom detector) -->
            <body name="head" pos="0 0.15 0.03">
                <joint name="head_pitch" type="hinge" axis="1 0 0" range="-0.5 0.8" damping="0.1"/>
                <geom name="head_geom" type="sphere" size="0.05" rgba="0.6 0.6 0.7 1" mass="0.2"/>
                <geom name="sensor_tip" type="sphere" pos="0 0.05 0.03" size="0.015" 
                      rgba="0.95 0.95 0.95 1"/>
                <site name="bloom_sensor" pos="0 0.05 0.03" size="0.015" rgba="0 1 0 0.5"/>
            </body>
            
            <!-- Left Arm (for tinkering) -->
            <body name="left_shoulder" pos="-0.08 0 0.03">
                <joint name="left_shoulder_yaw" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="0.05"/>
                <geom name="left_upper" type="capsule" fromto="0 0 0 -0.12 0 0" size="0.02" 
                      rgba="0.5 0.6 0.5 1" mass="0.15"/>
                
                <body name="left_elbow" pos="-0.12 0 0">
                    <joint name="left_elbow_pitch" type="hinge" axis="0 1 0" range="-2.0 0.5" damping="0.03"/>
                    <geom name="left_lower" type="capsule" fromto="0 0 0 -0.10 0 0" size="0.015" 
                          rgba="0.5 0.6 0.5 1" mass="0.1"/>
                    
                    <!-- Hand/Paddle for pushing -->
                    <body name="left_hand" pos="-0.10 0 0">
                        <joint name="left_wrist" type="hinge" axis="0 1 0" range="-1.0 1.0" damping="0.02"/>
                        <geom name="left_paddle" type="box" size="0.025 0.04 0.01" 
                              rgba="0.6 0.7 0.6 1" mass="0.05"/>
                    </body>
                </body>
            </body>
            
            <!-- Right Arm (for tinkering) -->
            <body name="right_shoulder" pos="0.08 0 0.03">
                <joint name="right_shoulder_yaw" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="0.05"/>
                <geom name="right_upper" type="capsule" fromto="0 0 0 0.12 0 0" size="0.02" 
                      rgba="0.5 0.6 0.5 1" mass="0.15"/>
                
                <body name="right_elbow" pos="0.12 0 0">
                    <joint name="right_elbow_pitch" type="hinge" axis="0 1 0" range="-0.5 2.0" damping="0.03"/>
                    <geom name="right_lower" type="capsule" fromto="0 0 0 0.10 0 0" size="0.015" 
                          rgba="0.5 0.6 0.5 1" mass="0.1"/>
                    
                    <body name="right_hand" pos="0.10 0 0">
                        <joint name="right_wrist" type="hinge" axis="0 1 0" range="-1.0 1.0" damping="0.02"/>
                        <geom name="right_paddle" type="box" size="0.025 0.04 0.01" 
                              rgba="0.6 0.7 0.6 1" mass="0.05"/>
                    </body>
                </body>
            </body>
        </body>
'''


def generate_leisure_env_xml(
    ceiling_radius=2.0,
    ceiling_height=1.2,
    sphere_spacing=0.08,  # Overlapping: spacing < 2*radius
    sphere_radius=0.06,
    num_blocks=8,
    output_path='sundog_leisure.xml'
):
    """
    Generate the complete leisure environment XML.
    """
    
    # Generate hexagonal ceiling lattice
    ceiling_positions = generate_hex_lattice(
        center_x=0, center_y=0,
        radius=ceiling_radius,
        spacing=sphere_spacing,
        z_base=ceiling_height,
        z_amplitude=0.02
    )
    
    # Generate play blocks
    blocks = generate_play_blocks(num_blocks=num_blocks, arena_radius=ceiling_radius * 0.6)
    
    # Build XML
    xml_parts = ['''<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="sundog_leisure">
    <compiler angle="radian" coordinate="local"/>
    <option gravity="0 0 -9.81" integrator="RK4" timestep="0.005"/>
    
    <default>
        <geom rgba="0.6 0.6 0.6 1" friction="0.8 0.4 0.001"/>
        <joint damping="0.1" armature="0.01"/>
    </default>
    
    <asset>
        <material name="floor_mat" rgba="0.3 0.35 0.3 1" specular="0.1" shininess="0.1"/>
        <material name="ceiling_mat" rgba="0.85 0.85 0.9 0.3" specular="0.3"/>
    </asset>

    <worldbody>
        <!-- Floor: soft surface for comfortable lying -->
        <geom name="floor" type="plane" pos="0 0 0" size="3 3 0.1" material="floor_mat"/>
        
        <!-- Arena boundary (soft walls) -->
        <geom name="wall_n" type="box" pos="0 2.5 0.3" size="2.5 0.1 0.3" rgba="0.4 0.4 0.45 0.5"/>
        <geom name="wall_s" type="box" pos="0 -2.5 0.3" size="2.5 0.1 0.3" rgba="0.4 0.4 0.45 0.5"/>
        <geom name="wall_e" type="box" pos="2.5 0 0.3" size="0.1 2.5 0.3" rgba="0.4 0.4 0.45 0.5"/>
        <geom name="wall_w" type="box" pos="-2.5 0 0.3" size="0.1 2.5 0.3" rgba="0.4 0.4 0.45 0.5"/>
        
        <!-- Ceiling base (transparent to see hexagonal lattice) -->
        <geom name="ceiling" type="box" pos="0 0 {ceiling_h}" size="2.5 2.5 0.005" 
              rgba="0.7 0.7 0.75 0.2" contype="0" conaffinity="0"/>
        
        <!-- Alignment target (the "song" point) -->
        <site name="resonance_point" pos="0 0 {ceiling_h}" size="0.03" rgba="1 0.8 0 0.8"/>
        
        <!-- Upward light source (creates shadows on hexagonal ceiling) -->
        <light name="bloom_light" pos="0 0 0.05" dir="0 0 1" directional="false"
               diffuse="0.8 0.7 0.6" specular="0.2 0.2 0.2" castshadow="true"
               cutoff="60" exponent="2"/>
        
        <!-- Ambient viewer light -->
        <light name="ambient" pos="0 0 3" dir="0 0 -1" directional="true"
               diffuse="0.4 0.4 0.4" castshadow="false"/>
        
        <!-- Cameras -->
        <camera name="top_down" pos="0 0 2.5" euler="0 0 0"/>
        <camera name="side_view" pos="2 0 0.5" euler="1.57 0 1.57"/>
        <camera name="agent_view" pos="0 -1.5 0.3" euler="1.4 0 0"/>
        
        <!-- Hexagonal Lattice Ceiling (overlapping spheres for bloom collapse) -->
        <body name="hex_ceiling">
'''.format(ceiling_h=ceiling_height)]
    
    # Add ceiling spheres
    for i, (x, y, z) in enumerate(ceiling_positions):
        xml_parts.append(f'            <geom type="sphere" pos="{x:.4f} {y:.4f} {z:.4f}" '
                        f'size="{sphere_radius}" rgba="0.85 0.85 0.9 0.6" '
                        f'contype="0" conaffinity="0"/>\n')
    
    xml_parts.append('        </body>\n\n')
    
    # Add play blocks (free bodies with mass)
    xml_parts.append('        <!-- Play Blocks (pushable, create shadows) -->\n')
    for block in blocks:
        xml_parts.append(f'''        <body name="{block['name']}" pos="{block['pos'][0]:.3f} {block['pos'][1]:.3f} {block['pos'][2]:.3f}">
            <freejoint name="{block['name']}_free"/>
            <geom type="box" size="{block['size'][0]} {block['size'][1]} {block['size'][2]}" 
                  rgba="{block['color']}" mass="{block['mass']}"/>
        </body>
''')
    
    # Add recumbent agent
    xml_parts.append('\n        <!-- Recumbent Agent -->\n')
    xml_parts.append(generate_recumbent_agent_xml())
    
    # Close worldbody and add actuators
    xml_parts.append('''
    </worldbody>

    <actuator>
        <!-- Head actuator for bloom sensing direction -->
        <motor joint="head_pitch" ctrlrange="-0.3 0.3" gear="2"/>
        
        <!-- Left arm actuators -->
        <motor joint="left_shoulder_yaw" ctrlrange="-0.5 0.5" gear="3"/>
        <motor joint="left_elbow_pitch" ctrlrange="-0.5 0.5" gear="2"/>
        <motor joint="left_wrist" ctrlrange="-0.3 0.3" gear="1"/>
        
        <!-- Right arm actuators -->
        <motor joint="right_shoulder_yaw" ctrlrange="-0.5 0.5" gear="3"/>
        <motor joint="right_elbow_pitch" ctrlrange="-0.5 0.5" gear="2"/>
        <motor joint="right_wrist" ctrlrange="-0.3 0.3" gear="1"/>
    </actuator>
    
    <sensor>
        <!-- Proprioceptive sensors (lightweight) -->
        <jointpos joint="head_pitch" name="head_pos"/>
        <jointpos joint="left_shoulder_yaw" name="l_shoulder_pos"/>
        <jointpos joint="left_elbow_pitch" name="l_elbow_pos"/>
        <jointpos joint="right_shoulder_yaw" name="r_shoulder_pos"/>
        <jointpos joint="right_elbow_pitch" name="r_elbow_pos"/>
        
        <!-- Torque sensors for indirect feedback -->
        <torque joint="left_elbow_pitch" name="l_elbow_torque"/>
        <torque joint="right_elbow_pitch" name="r_elbow_torque"/>
        
        <!-- Site position for bloom tracking -->
        <framepos objtype="site" objname="bloom_sensor" name="bloom_pos"/>
    </sensor>
</mujoco>
''')
    
    xml_content = ''.join(xml_parts)
    
    with open(output_path, 'w') as f:
        f.write(xml_content)
    
    print(f"Generated {output_path}")
    print(f"  - Ceiling spheres: {len(ceiling_positions)}")
    print(f"  - Play blocks: {len(blocks)}")
    print(f"  - Agent DOF: 7 (head + 2x arm)")
    
    return xml_content


if __name__ == '__main__':
    generate_leisure_env_xml(
        ceiling_radius=1.8,
        ceiling_height=1.0,
        sphere_spacing=0.07,
        sphere_radius=0.05,
        num_blocks=10,
        output_path='sundog_leisure.xml'
    )