import torch
import torch.nn as nn
import torch.nn.functional as F

class KarmicAgent(nn.Module):
    """
    A Unified Agent Architecture for Karmic-RL.
    
    Modes:
    1. Baseline (use_tvt=False): Standard DRQN (CNN + LSTM + Actor/Critic).
    2. Karmic-TVT (use_tvt=True): Adds External Memory and Attentional Read Head.
    
    The 'Semantic Encoder' is always present but only used for TVT queries
    if TVT is enabled.
    """
    def __init__(self, obs_shape, action_dim, hidden_dim=256, use_tvt=False, memory_size=1000):
        super(KarmicAgent, self).__init__()
        self.use_tvt = use_tvt
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.memory_size = memory_size

        # --- 1. Visual Encoder (CNN) ---
        # Input: (Batch, Channels, Height, Width) -> e.g. (B, 3, 20, 20)
        c, h, w = obs_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            self.cnn_out_dim = self.cnn(dummy).shape[1]

        # --- 2. Core Recurrent Brain (LSTM) ---
        # Takes CNN features + (Optional) Memory Read
        input_dim = self.cnn_out_dim
        if self.use_tvt:
            input_dim += hidden_dim  # We concatenate the retrieved memory vector

        self.lstm = nn.LSTMCell(input_dim, hidden_dim)

        # --- 3. Action & Value Heads (PPO/A2C) ---
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        # --- 4. TVT Components (The "Time Machine") ---
        if self.use_tvt:
            # Semantic Encoder: Predicts 'Social Context' from visual features
            # This is trained via auxiliary loss to output role (Aggressor/Victim/Neutral)
            self.context_encoder = nn.Linear(self.cnn_out_dim, 32)
            
            # Key/Query Generators
            # Query: Generated from current state (Context) -> "What happened before?"
            self.query_net = nn.Linear(hidden_dim, 32)
            # Key: Generated from past states -> "I was doing X"
            self.key_net = nn.Linear(hidden_dim, 32)

            # The Memory Bank (Not a Parameter, but a Buffer)
            # Stores [Key, Value] pairs.
            # Value = The LSTM hidden state at that time.
            self.register_buffer("memory_keys", torch.zeros(1, memory_size, 32))
            self.register_buffer("memory_values", torch.zeros(1, memory_size, hidden_dim))
            self.register_buffer("mem_ptr", torch.zeros(1, dtype=torch.long)) # Current write pointer

    def forward(self, obs, lstm_state, done_mask=None):
        """
        Forward pass for one timestep.
        
        obs: (Batch, C, H, W)
        lstm_state: (hx, cx) tuple or None
        done_mask: (Batch, 1) - 1.0 if episode ended, to reset memory/state
        """
        batch_size = obs.size(0)

        # 1. Vision
        visual_feats = self.cnn(obs)

        # 2. TVT Logic (Read from Past)
        memory_read = None
        if self.use_tvt:
            # Create Query based on current LSTM hidden state (Short-term context)
            # Note: In a real implementation, we might use the PREVIOUS hidden state
            # or the visual features to form the query before the LSTM update.
            # Here, we use visual features for the query to "look before we leap"
            query = self.context_encoder(visual_feats) # Using Semantic Context as query
            
            # Read from External Memory
            # (Batch, 1, 32) x (Batch, 32, MemSize) -> (Batch, 1, MemSize)
            scores = torch.bmm(query.unsqueeze(1), self.memory_keys.transpose(1, 2))
            attn_weights = F.softmax(scores, dim=-1)

            # Weighted Sum of Values
            # (Batch, 1, MemSize) x (Batch, MemSize, Hidden) -> (Batch, 1, Hidden)
            memory_read = torch.bmm(attn_weights, self.memory_values).squeeze(1)
            
            # Combine Vision + Memory
            lstm_input = torch.cat([visual_feats, memory_read], dim=1)
        else:
            lstm_input = visual_feats

        # 3. Update Brain (LSTM)
        if lstm_state is None:
            # Fix: Handle None state (e.g. during training updates)
            hx, cx = self.get_initial_state(batch_size)
            hx = hx.to(visual_feats.device)
            cx = cx.to(visual_feats.device)
        else:
            hx, cx = lstm_state

        if done_mask is not None:
            # Ensure mask is (B, 1) to broadcast correctly across hidden dim (B, H)
            mask_expanded = done_mask.view(-1, 1)
            hx = hx * (1 - mask_expanded)
            cx = cx * (1 - mask_expanded)
            
            # Also reset memory pointer for that batch item (simplified logic here)
            if done_mask.sum() > 0:
                self.reset_memory()

        new_hx, new_cx = self.lstm(lstm_input, (hx, cx))

        # 4. TVT Logic (Write to Past)
        if self.use_tvt:
            self._write_to_memory(new_hx, visual_feats)

        # 5. Output Actions
        logits = self.actor(new_hx)
        value = self.critic(new_hx)

        # Return extra info for training (e.g. attention weights for visualization)
        extras = {"attn": attn_weights} if self.use_tvt else {}

        return logits, value, (new_hx, new_cx), extras

    def _write_to_memory(self, hidden_state, visual_feats):
        """
        Stores the current experience into the circular buffer.
        Key = Semantic Context (What was I doing?)
        Value = Hidden State (What was I thinking?)
        """
        batch_size = hidden_state.size(0)
        ptr = int(self.mem_ptr.item())

        # Generate Key from the encoded context
        key = self.context_encoder(visual_feats)

        # Write (Simplified: assumes batch_size=1 for clarity, 
        # real implementation handles batched pointers)
        if ptr < self.memory_size:
            self.memory_keys[:, ptr] = key
            self.memory_values[:, ptr] = hidden_state

        # Circular Buffer Logic
        new_ptr = (ptr + 1) % self.memory_size
        self.mem_ptr[0] = new_ptr

    def reset_memory(self):
        """Clears the external memory (e.g. on episode start)."""
        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.mem_ptr.zero_()

    def get_initial_state(self, batch_size):
        return (
            torch.zeros(batch_size, self.hidden_dim),
            torch.zeros(batch_size, self.hidden_dim)
        )
