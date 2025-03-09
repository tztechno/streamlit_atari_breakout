import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageDraw

# Simple Breakout game implementation
class SimpleBreakout:
    def __init__(self, width=400, height=600):
        self.width = width
        self.height = height
        self.paddle_width = 80
        self.paddle_height = 10
        self.ball_radius = 8
        self.brick_rows = 4
        self.brick_cols = 8
        self.brick_width = (width - 40) // self.brick_cols
        self.brick_height = 20
        self.reset()
    
    def reset(self):
        # Initialize game state
        self.score = 0
        self.lives = 5
        self.game_over = False
        
        # Initialize paddle
        self.paddle_x = self.width // 2 - self.paddle_width // 2
        self.paddle_y = self.height - 40
        
        # Initialize ball
        self.ball_x = self.width // 2
        self.ball_y = self.height - 60
        self.ball_dx = 4
        self.ball_dy = -4
        
        # Initialize bricks
        self.bricks = []
        for row in range(self.brick_rows):
            for col in range(self.brick_cols):
                self.bricks.append({
                    'x': col * self.brick_width + 20,
                    'y': row * self.brick_height + 50,
                    'visible': True
                })
        
        return self.get_frame()
    
    def step(self, action):
        # 0: no action, 1: fire, 2: move right, 3: move left
        
        # Move paddle
        if action == 2:  # RIGHT
            self.paddle_x = min(self.paddle_x + 10, self.width - self.paddle_width)
        elif action == 3:  # LEFT
            self.paddle_x = max(self.paddle_x - 10, 0)
        
        # Move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # Ball collision with walls
        if self.ball_x <= 0 or self.ball_x >= self.width:
            self.ball_dx *= -1
        if self.ball_y <= 0:
            self.ball_dy *= -1
        
        # Ball collision with paddle
        if (self.ball_y + self.ball_radius >= self.paddle_y and 
            self.ball_x >= self.paddle_x and 
            self.ball_x <= self.paddle_x + self.paddle_width):
            self.ball_dy *= -1
            # Change angle based on where the ball hit the paddle
            paddle_middle = self.paddle_x + self.paddle_width // 2
            offset = (self.ball_x - paddle_middle) / (self.paddle_width // 2)
            self.ball_dx = offset * 5
        
        # Ball collision with bricks
        for brick in self.bricks:
            if (brick['visible'] and 
                self.ball_x >= brick['x'] and 
                self.ball_x <= brick['x'] + self.brick_width and 
                self.ball_y >= brick['y'] and 
                self.ball_y <= brick['y'] + self.brick_height):
                brick['visible'] = False
                self.score += 10
                self.ball_dy *= -1
                break
        
        # Ball out of bounds
        if self.ball_y > self.height:
            self.lives -= 1
            self.ball_x = self.width // 2
            self.ball_y = self.height - 60
            self.ball_dx = 4
            self.ball_dy = -4
            if self.lives <= 0:
                self.game_over = True
        
        # Check if all bricks are gone
        if all(not brick['visible'] for brick in self.bricks):
            # Add new row of bricks
            for col in range(self.brick_cols):
                self.bricks.append({
                    'x': col * self.brick_width + 20,
                    'y': 50,
                    'visible': True
                })
            # Move all bricks down
            for brick in self.bricks:
                brick['y'] += self.brick_height
        
        # Get the current frame
        frame = self.get_frame()
        
        # Return the observation, reward, done, and info
        reward = 0.1  # Small reward for staying alive
        info = {'ale.lives': self.lives}
        
        return frame, reward, self.game_over, info
    
    def get_frame(self):
        # Create a blank image
        img = Image.new('RGB', (self.width, self.height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw paddle
        draw.rectangle([self.paddle_x, self.paddle_y, 
                        self.paddle_x + self.paddle_width, self.paddle_y + self.paddle_height], 
                       fill=(200, 200, 200))
        
        # Draw ball
        draw.ellipse([self.ball_x - self.ball_radius, self.ball_y - self.ball_radius,
                      self.ball_x + self.ball_radius, self.ball_y + self.ball_radius],
                     fill=(255, 255, 255))
        
        # Draw bricks
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for i, brick in enumerate(self.bricks):
            if brick['visible']:
                row = i // self.brick_cols
                color = colors[row % len(colors)]
                draw.rectangle([brick['x'], brick['y'],
                                brick['x'] + self.brick_width - 2, brick['y'] + self.brick_height - 2],
                               fill=color)
        
        # Convert to numpy array
        return np.array(img)

# Function to convert slider value to action
def get_action_from_slider(slider_value):
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
    st.title("Play Breakout Manually")
    st.markdown("""
    Control the paddle with the slider below. 
    - Left side of slider: Move paddle left
    - Right side of slider: Move paddle right
    - Middle: No movement
    - Press 'Fire Ball' button to start a new game
    """)
    
    # Session state to keep track of game state
    if 'env' not in st.session_state:
        st.session_state.env = SimpleBreakout()
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
        game_image.image(frame, caption="Breakout Game", use_column_width=True)
    
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
            obs, reward, done, info = st.session_state.env.step(action)
            
            # Update session state
            st.session_state.obs = obs
            st.session_state.total_reward += reward
            st.session_state.done = done
            
            # Update game display
            game_image.image(obs, caption="Breakout Game", use_column_width=True)
            
            # Update stats
            score_text.markdown(f"**Score:** {int(st.session_state.total_reward)}")
            
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

