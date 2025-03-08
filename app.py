import streamlit as st
import numpy as np
import gym  # Using gym instead of gymnasium
from gym.wrappers import AtariPreprocessing  # Older gym API
import time
import matplotlib.pyplot as plt
from PIL import Image

def create_breakout_env():
    """Create Atari Breakout environment with basic preprocessing"""
    env = gym.make('BreakoutNoFrameskip-v4', render_mode='rgb_array')
    # Basic preprocessing
    env = AtariPreprocessing(
        env, 
        frame_skip=4,
        grayscale_obs=False,  # Keep color for better visualization
        scale_obs=False,
        terminal_on_life_loss=False
    )
    return env

def get_action_from_slider(slider_value, action_space_size=4):
    """Convert slider value (0-100) to action (0, 1, 2, 3)"""
    # Map slider value (0-100) to action space
    # For Breakout: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
    if slider_value < 25:
        return 3  # LEFT - strong
    elif slider_value < 45:
        return 3  # LEFT - mild
    elif slider_value < 55:
        return 0  # NOOP (middle position)
    elif slider_value < 75:
        return 2  # RIGHT - mild
    else:
        return 2  # RIGHT - strong

def main():
    st.title("Play Atari Breakout Manually")
    st.markdown("""
    Control the paddle with the slider below. 
    - Left side of slider: Move paddle left
    - Right side of slider: Move paddle right
    - Middle: No movement
    - Press 'Fire Ball' button to start a new game
    """)
    
    # Session state to keep track of game state
    if 'env' not in st.session_state:
        st.session_state.env = create_breakout_env()
        st.session_state.obs = st.session_state.env.reset()
        st.session_state.total_reward = 0
        st.session_state.done = False
        st.session_state.episode_rewards = []
        st.session_state.current_lives = 5
    
    # Create columns for game display and stats
    col1, col2 = st.columns([3, 1])
    
    # Game display
    with col1:
        # Display current frame
        game_image = st.empty()
        frame = st.session_state.obs
        if len(frame.shape) == 3:  # RGB frame
            game_image.image(frame, caption="Breakout Game", use_column_width=True)
        else:  # Grayscale frame
            game_image.image(frame, caption="Breakout Game", use_column_width=True, channels="GRAY")
    
    # Game stats
    with col2:
        st.subheader("Game Stats")
        score_text = st.empty()
        lives_text = st.empty()
        score_text.markdown(f"**Score:** {st.session_state.total_reward}")
        lives_text.markdown(f"**Lives:** {st.session_state.current_lives}")
        
        # Add Fire button to start new game
        if st.button("Fire Ball"):
            # Fire action (1) to start game or launch new ball
            temp_obs, temp_reward, temp_done, temp_info = st.session_state.env.step(1)
            st.session_state.obs = temp_obs
            # Update the game display
            game_image.image(temp_obs, caption="Breakout Game", use_column_width=True)
    
    # Control slider
    st.subheader("Paddle Control")
    paddle_position = st.slider("Move Paddle", 0, 100, 50, key="paddle_slider")
    
    # Game step logic
    if not st.session_state.done:
        # Get action from slider
        action = get_action_from_slider(paddle_position)
        
        # Take action in environment
        try:
            # Old gym API
            obs, reward, done, info = st.session_state.env.step(action)
            
            # Update session state
            st.session_state.obs = obs
            st.session_state.total_reward += reward
            st.session_state.done = done
            
            # Update game display
            game_image.image(obs, caption="Breakout Game", use_column_width=True)
            
            # Update stats
            score_text.markdown(f"**Score:** {st.session_state.total_reward}")
            
            # Check for lives change
            if 'ale.lives' in info and info['ale.lives'] != st.session_state.current_lives:
                st.session_state.current_lives = info['ale.lives']
                lives_text.markdown(f"**Lives:** {st.session_state.current_lives}")
        
        except Exception as e:
            st.error(f"Error during game step: {e}")
    
    # Reset if game is over
    if st.session_state.done:
        st.warning("Game Over! Press Reset to play again.")
        st.session_state.episode_rewards.append(st.session_state.total_reward)
        
        if st.button("Reset Game"):
            st.session_state.obs = st.session_state.env.reset()
            st.session_state.total_reward = 0
            st.session_state.done = False
            st.session_state.current_lives = 5
            st.experimental_rerun()
    
    # Display historical performance
    if len(st.session_state.episode_rewards) > 0:
        st.subheader("Your Performance History")
        fig, ax = plt.subplots()
        ax.plot(st.session_state.episode_rewards)
        ax.set_xlabel("Games")
        ax.set_ylabel("Score")
        ax.set_title("Score History")
        st.pyplot(fig)
    
    # Add some instructions and controls
    st.markdown("---")
    st.subheader("Instructions")
    st.markdown("""
    - The slider controls the paddle's position
    - Press 'Fire Ball' to start a new ball
    - Try to break all the bricks without losing your ball
    - You have 5 lives per game
    """)
    
    # Add a small delay to make the game playable (otherwise it runs too fast)
    time.sleep(0.05)
    st.experimental_rerun()

if __name__ == "__main__":
    main()
