use brush_serde::ProgressiveShCheckpointMetadata;

pub const PROGRESSIVE_SH_SCHEDULE_VERSION: u32 = 1;
pub const PROGRESSIVE_SH_MODE_DISABLED: u32 = 0;
pub const PROGRESSIVE_SH_MODE_INTERVAL: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShTransitionReason {
    Disabled,
    Initial,
    Scheduled,
    Resume,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgressiveShSchedule {
    mode: u32,
    maximum_degree: u32,
    initial_degree: u32,
    step_interval: u32,
    total_iterations: u32,
    transitions: Vec<(u32, u32)>,
    identity: u64,
}

impl ProgressiveShSchedule {
    pub fn new(
        mode: u32,
        maximum_degree: u32,
        initial_degree: u32,
        step_interval: u32,
        total_iterations: u32,
    ) -> Result<Self, String> {
        if maximum_degree > 4 {
            return Err(format!(
                "progressive SH maximum degree {maximum_degree} exceeds supported degree 4"
            ));
        }
        if total_iterations == 0 {
            return Err("progressive SH total iterations must be non-zero".to_owned());
        }

        let transitions = match mode {
            PROGRESSIVE_SH_MODE_DISABLED => {
                if initial_degree != maximum_degree || step_interval != 0 {
                    return Err(
                        "disabled progressive SH requires initial degree equal to maximum and interval 0"
                            .to_owned(),
                    );
                }
                vec![(0, maximum_degree)]
            }
            PROGRESSIVE_SH_MODE_INTERVAL => {
                if initial_degree >= maximum_degree {
                    return Err(
                        "interval progressive SH requires initial degree below maximum".to_owned(),
                    );
                }
                if step_interval == 0 {
                    return Err("interval progressive SH requires a non-zero interval".to_owned());
                }
                let mut transitions =
                    Vec::with_capacity((maximum_degree - initial_degree + 1) as usize);
                for degree in initial_degree..=maximum_degree {
                    let offset = degree - initial_degree;
                    let iteration = offset.checked_mul(step_interval).ok_or_else(|| {
                        "progressive SH transition iteration overflowed".to_owned()
                    })?;
                    if iteration >= total_iterations {
                        return Err(format!(
                            "progressive SH transition for degree {degree} at iteration {iteration} is outside total iterations {total_iterations}"
                        ));
                    }
                    transitions.push((iteration, degree));
                }
                transitions
            }
            other => return Err(format!("unsupported progressive SH schedule mode {other}")),
        };

        for pair in transitions.windows(2) {
            let [(prior_iteration, prior_degree), (iteration, degree)] = pair else {
                unreachable!("windows(2) always yields two entries")
            };
            if iteration <= prior_iteration || degree <= prior_degree {
                return Err(
                    "progressive SH transitions must be unique and strictly increasing".to_owned(),
                );
            }
        }

        let identity = schedule_identity(
            mode,
            maximum_degree,
            initial_degree,
            step_interval,
            total_iterations,
            &transitions,
        );
        Ok(Self {
            mode,
            maximum_degree,
            initial_degree,
            step_interval,
            total_iterations,
            transitions,
            identity,
        })
    }

    pub const fn mode(&self) -> u32 {
        self.mode
    }

    pub const fn maximum_degree(&self) -> u32 {
        self.maximum_degree
    }

    pub const fn initial_degree(&self) -> u32 {
        self.initial_degree
    }

    pub const fn step_interval(&self) -> u32 {
        self.step_interval
    }

    pub const fn total_iterations(&self) -> u32 {
        self.total_iterations
    }

    pub const fn identity(&self) -> u64 {
        self.identity
    }

    pub fn transitions(&self) -> &[(u32, u32)] {
        &self.transitions
    }

    pub fn active_degree_at(&self, zero_based_iteration: u32) -> u32 {
        self.transitions
            .iter()
            .rev()
            .find_map(|(iteration, degree)| (zero_based_iteration >= *iteration).then_some(*degree))
            .unwrap_or(self.initial_degree)
    }

    pub fn transition_degree_at(&self, zero_based_iteration: u32) -> Option<u32> {
        self.transitions
            .iter()
            .find_map(|(iteration, degree)| (*iteration == zero_based_iteration).then_some(*degree))
    }

    pub fn checkpoint_metadata(
        &self,
        completed_iterations: u32,
    ) -> ProgressiveShCheckpointMetadata {
        let last_iteration = completed_iterations.saturating_sub(1);
        ProgressiveShCheckpointMetadata {
            version: PROGRESSIVE_SH_SCHEDULE_VERSION,
            schedule_identity: self.identity,
            schedule_mode: self.mode,
            maximum_degree: self.maximum_degree,
            initial_degree: self.initial_degree,
            step_interval: self.step_interval,
            total_iterations: self.total_iterations,
            completed_iterations,
            active_degree: self.active_degree_at(last_iteration),
        }
    }

    pub fn validate_checkpoint(
        &self,
        checkpoint: &ProgressiveShCheckpointMetadata,
        expected_completed_iterations: u32,
    ) -> Result<(), String> {
        if checkpoint.version != PROGRESSIVE_SH_SCHEDULE_VERSION
            || checkpoint.schedule_identity != self.identity
            || checkpoint.schedule_mode != self.mode
            || checkpoint.maximum_degree != self.maximum_degree
            || checkpoint.initial_degree != self.initial_degree
            || checkpoint.step_interval != self.step_interval
            || checkpoint.total_iterations != self.total_iterations
        {
            return Err("checkpoint progressive SH schedule identity is incompatible".to_owned());
        }
        if checkpoint.completed_iterations != expected_completed_iterations {
            return Err(format!(
                "checkpoint completed iteration count {} does not match requested start iteration {expected_completed_iterations}",
                checkpoint.completed_iterations
            ));
        }
        let expected_active =
            self.active_degree_at(expected_completed_iterations.saturating_sub(1));
        if checkpoint.active_degree != expected_active {
            return Err(format!(
                "checkpoint active SH degree {} does not match expected completed-step degree {expected_active}",
                checkpoint.active_degree
            ));
        }
        Ok(())
    }
}

fn schedule_identity(
    mode: u32,
    maximum_degree: u32,
    initial_degree: u32,
    step_interval: u32,
    total_iterations: u32,
    transitions: &[(u32, u32)],
) -> u64 {
    // Stable FNV-1a over explicitly little-endian schedule fields. This is a
    // configuration identity, not a cryptographic integrity primitive.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in [
        PROGRESSIVE_SH_SCHEDULE_VERSION,
        mode,
        maximum_degree,
        initial_degree,
        step_interval,
        total_iterations,
        transitions.len() as u32,
    ] {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    for (iteration, degree) in transitions {
        for byte in iteration
            .to_le_bytes()
            .into_iter()
            .chain(degree.to_le_bytes())
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phase_4_3() -> ProgressiveShSchedule {
        ProgressiveShSchedule::new(PROGRESSIVE_SH_MODE_INTERVAL, 3, 0, 150, 600).unwrap()
    }

    #[test]
    fn phase_4_3_boundaries_are_exact() {
        let schedule = phase_4_3();
        assert_eq!(
            schedule.transitions(),
            &[(0, 0), (150, 1), (300, 2), (450, 3)]
        );
        for (iteration, expected) in [
            (0, 0),
            (149, 0),
            (150, 1),
            (299, 1),
            (300, 2),
            (449, 2),
            (450, 3),
            (599, 3),
        ] {
            assert_eq!(schedule.active_degree_at(iteration), expected);
        }
    }

    #[test]
    fn resume_uses_next_iteration_without_double_transition() {
        let schedule = phase_4_3();
        let before = schedule.checkpoint_metadata(149);
        schedule.validate_checkpoint(&before, 149).unwrap();
        assert_eq!(before.active_degree, 0);
        assert_eq!(schedule.active_degree_at(149), 0);

        let boundary = schedule.checkpoint_metadata(150);
        schedule.validate_checkpoint(&boundary, 150).unwrap();
        assert_eq!(boundary.active_degree, 0);
        assert_eq!(schedule.active_degree_at(150), 1);

        let after = schedule.checkpoint_metadata(151);
        schedule.validate_checkpoint(&after, 151).unwrap();
        assert_eq!(after.active_degree, 1);
        assert_eq!(schedule.active_degree_at(151), 1);
    }

    #[test]
    fn invalid_schedules_fail_closed() {
        assert!(ProgressiveShSchedule::new(99, 3, 0, 150, 600).is_err());
        assert!(ProgressiveShSchedule::new(PROGRESSIVE_SH_MODE_DISABLED, 3, 0, 0, 600).is_err());
        assert!(ProgressiveShSchedule::new(PROGRESSIVE_SH_MODE_INTERVAL, 3, 3, 150, 600).is_err());
        assert!(ProgressiveShSchedule::new(PROGRESSIVE_SH_MODE_INTERVAL, 3, 0, 0, 600).is_err());
        assert!(ProgressiveShSchedule::new(PROGRESSIVE_SH_MODE_INTERVAL, 3, 0, 200, 600).is_err());
    }

    #[test]
    fn checkpoint_identity_rejects_mismatch() {
        let schedule = phase_4_3();
        let mut checkpoint = schedule.checkpoint_metadata(300);
        checkpoint.schedule_identity ^= 1;
        assert!(schedule.validate_checkpoint(&checkpoint, 300).is_err());
    }
}
