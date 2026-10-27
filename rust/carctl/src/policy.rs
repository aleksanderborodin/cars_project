//! Controller-only edit surface. ABI v1, all state/track numbers are f32.
//! Track stride >= 4: x, y, tangent heading, half-width (metres/radians).
//! State (20 floats), world frame, heading forward from +x, CCW positive:
//! 0 x, 1 y, 2 heading, 3 vx, 4 vy, 5 yaw_rate;
//! 6..10 wheel omega; 10..14 actual wheel steer angle;
//! 14..18 wheel on-road flags (0/1); 18 nearest tile index; 19 visited fraction.
//! Wheels: front-left, front-right, rear-left, rear-right.
//! Output: steer [-1,1] (positive RIGHT), gas [0,1], brake [0,1].
//! reset receives a copied track owned by this controller. act's input is valid
//! only for that call. Never retain input pointers. No dependencies or RNG.

#[derive(Default)]
pub struct Controller {
    track: Vec<[f32; 4]>,
}

impl Controller {
    pub fn reset(&mut self, track: Vec<[f32; 4]>) {
        self.track = track;
    }

    pub fn act(&mut self, state: &[f32]) -> [f32; 3] {
        if self.track.is_empty() || state.len() < 20 {
            return [0.0, 0.0, 0.0];
        }
        let idx = (state[18].max(0.0) as usize + 3) % self.track.len();
        let dx = self.track[idx][0] - state[0];
        let dy = self.track[idx][1] - state[1];
        let angle = dy.atan2(dx) - state[2];
        let error = angle.sin().atan2(angle.cos());
        [(-error).clamp(-1.0, 1.0), 0.3, 0.0]
    }
}
