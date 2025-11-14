#!/usr/bin/env python3
"""
SVD Metrics Monitoring Script

Monitors adaptive SVD training in real-time and alerts when mode switch is imminent.

Usage:
    # Monitor WandB run in real-time
    python scripts/monitor_svd_metrics.py --wandb-run <run_id>

    # Monitor training output log file
    python scripts/monitor_svd_metrics.py --log-file <path_to_log>

    # Monitor WandB with custom thresholds
    python scripts/monitor_svd_metrics.py --wandb-run <run_id> --warn-margin 0.05
"""

import argparse
import time
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


@dataclass
class SVDMetrics:
    """Container for SVD metrics at a given step."""
    step: int
    layer_name: str
    mode: float  # 0.0 = full, 1.0 = lowrank
    r: Optional[int]
    principal_alignment: Optional[float]
    minor_alignment: Optional[float]
    subspace_angle: Optional[float]
    reconstruction_error: Optional[float]

    def __str__(self) -> str:
        mode_str = "LOW-RANK" if self.mode == 1.0 else "FULL"
        return (f"Step {self.step} [{self.layer_name}] Mode={mode_str} r={self.r} | "
                f"principal={self.principal_alignment:.3f} minor={self.minor_alignment:.3f} "
                f"angle={self.subspace_angle:.3f} error={self.reconstruction_error:.4f}")


@dataclass
class SwitchThresholds:
    """Decision thresholds for mode switching."""
    # Immediate clobbering trigger
    principal_alignment_danger: float = 0.4

    # Optimal conditions trigger (all must be true)
    subspace_angle_stable: float = 0.1
    minor_alignment_safe: float = 0.6
    reconstruction_error_safe: float = 0.01

    # Warning thresholds (alert when approaching)
    warn_margin: float = 0.05  # How close to trigger before warning


class SVDMonitor:
    """Monitors SVD metrics and alerts on imminent mode switches."""

    def __init__(self, thresholds: SwitchThresholds, history_size: int = 20):
        self.thresholds = thresholds
        self.history: Dict[str, deque] = {}  # layer_name -> deque of SVDMetrics
        self.history_size = history_size
        self.last_alert_step: Dict[str, int] = {}  # Avoid duplicate alerts

    def add_metrics(self, metrics: SVDMetrics):
        """Add new metrics and check for alerts."""
        layer = metrics.layer_name

        # Initialize history for this layer if needed
        if layer not in self.history:
            self.history[layer] = deque(maxlen=self.history_size)

        self.history[layer].append(metrics)

        # Check for alerts
        self._check_alerts(metrics)

    def _check_alerts(self, m: SVDMetrics):
        """Check if metrics warrant an alert."""
        layer = m.layer_name

        # Skip if we already alerted for this step
        if self.last_alert_step.get(layer) == m.step:
            return

        alerts = []

        # Check for immediate clobbering danger
        if m.principal_alignment is not None:
            if m.principal_alignment >= self.thresholds.principal_alignment_danger:
                alerts.append((
                    "CRITICAL",
                    f"CLOBBERING DETECTED! principal_alignment={m.principal_alignment:.3f} >= {self.thresholds.principal_alignment_danger}"
                ))
            elif m.principal_alignment >= self.thresholds.principal_alignment_danger - self.thresholds.warn_margin:
                distance = self.thresholds.principal_alignment_danger - m.principal_alignment
                alerts.append((
                    "WARNING",
                    f"Approaching clobbering threshold: principal_alignment={m.principal_alignment:.3f} (within {distance:.3f})"
                ))

        # Check for optimal conditions approaching
        conditions = []
        approaching = []

        if m.subspace_angle is not None:
            if m.subspace_angle < self.thresholds.subspace_angle_stable:
                conditions.append("✓ Stable subspace")
            elif m.subspace_angle < self.thresholds.subspace_angle_stable + self.thresholds.warn_margin:
                distance = m.subspace_angle - self.thresholds.subspace_angle_stable
                approaching.append(f"subspace_angle={m.subspace_angle:.3f} (within {distance:.3f} of 0.1)")

        if m.minor_alignment is not None:
            if m.minor_alignment > self.thresholds.minor_alignment_safe:
                conditions.append("✓ Safe gradients")
            elif m.minor_alignment > self.thresholds.minor_alignment_safe - self.thresholds.warn_margin:
                distance = self.thresholds.minor_alignment_safe - m.minor_alignment
                approaching.append(f"minor_alignment={m.minor_alignment:.3f} (within {distance:.3f} of 0.6)")

        if m.reconstruction_error is not None:
            if m.reconstruction_error < self.thresholds.reconstruction_error_safe:
                conditions.append("✓ Safe reconstruction")
            # Note: reconstruction_error can be very small, so we don't warn about it

        # Alert if all conditions met
        if len(conditions) == 3:
            alerts.append((
                "CRITICAL",
                f"OPTIMAL CONDITIONS MET! All criteria satisfied → SWITCH IMMINENT"
            ))
        # Alert if 2/3 conditions met and third is approaching
        elif len(conditions) >= 1 and len(approaching) >= 1:
            alerts.append((
                "WARNING",
                f"Approaching optimal conditions: {', '.join(conditions + approaching)}"
            ))

        # Check for trend (getting closer to switch over last N steps)
        trend = self._analyze_trend(layer)
        if trend:
            alerts.append(("INFO", trend))

        # Print alerts
        if alerts:
            self._print_alert(m, alerts)
            self.last_alert_step[layer] = m.step

    def _analyze_trend(self, layer: str) -> Optional[str]:
        """Analyze trend over recent history."""
        if layer not in self.history or len(self.history[layer]) < 5:
            return None

        history = list(self.history[layer])
        recent = history[-5:]

        # Check if subspace_angle is decreasing
        if all(m.subspace_angle is not None for m in recent):
            angles = [m.subspace_angle for m in recent]
            if all(angles[i] > angles[i+1] for i in range(len(angles)-1)):
                rate = (angles[0] - angles[-1]) / len(angles)
                steps_to_threshold = (angles[-1] - self.thresholds.subspace_angle_stable) / rate if rate > 0 else float('inf')
                if 0 < steps_to_threshold < 100:
                    return f"Subspace stabilizing: {angles[-1]:.3f} → ~{int(steps_to_threshold)} steps to threshold"

        # Check if minor_alignment is increasing
        if all(m.minor_alignment is not None for m in recent):
            aligns = [m.minor_alignment for m in recent]
            if all(aligns[i] < aligns[i+1] for i in range(len(aligns)-1)):
                rate = (aligns[-1] - aligns[0]) / len(aligns)
                steps_to_threshold = (self.thresholds.minor_alignment_safe - aligns[-1]) / rate if rate > 0 else float('inf')
                if 0 < steps_to_threshold < 100:
                    return f"Gradients becoming safer: {aligns[-1]:.3f} → ~{int(steps_to_threshold)} steps to threshold"

        return None

    def _print_alert(self, m: SVDMetrics, alerts: List[Tuple[str, str]]):
        """Print formatted alert."""
        print("\n" + "="*80)
        print(f"{Colors.BOLD}Step {m.step} | {m.layer_name}{Colors.ENDC}")
        print("-"*80)

        for level, message in alerts:
            if level == "CRITICAL":
                color = Colors.FAIL
                prefix = "🚨 CRITICAL"
            elif level == "WARNING":
                color = Colors.WARNING
                prefix = "⚠️  WARNING"
            else:
                color = Colors.OKCYAN
                prefix = "ℹ️  INFO"

            print(f"{color}{prefix}: {message}{Colors.ENDC}")

        # Print current metrics
        print("\nCurrent metrics:")
        print(f"  principal_alignment: {m.principal_alignment:.4f} (danger at >{self.thresholds.principal_alignment_danger})")
        print(f"  minor_alignment:     {m.minor_alignment:.4f} (safe at >{self.thresholds.minor_alignment_safe})")
        print(f"  subspace_angle:      {m.subspace_angle:.4f} (stable at <{self.thresholds.subspace_angle_stable})")
        print(f"  reconstruction_error:{m.reconstruction_error:.4f} (safe at <{self.thresholds.reconstruction_error_safe})")
        print(f"  mode:                {'LOW-RANK' if m.mode == 1.0 else 'FULL'}")
        if m.r is not None:
            print(f"  r:                   {m.r}")

        print("="*80 + "\n")

    def print_summary(self):
        """Print summary of all monitored layers."""
        if not self.history:
            print("No metrics collected yet.")
            return

        print("\n" + "="*80)
        print(f"{Colors.BOLD}SVD MONITORING SUMMARY{Colors.ENDC}")
        print("="*80)

        for layer, metrics_history in self.history.items():
            latest = metrics_history[-1]
            print(f"\n{Colors.HEADER}{layer}{Colors.ENDC}")
            print(f"  Latest step: {latest.step}")
            print(f"  Mode: {'LOW-RANK' if latest.mode == 1.0 else 'FULL'}")
            if latest.r is not None:
                print(f"  Rank: {latest.r}")
            print(f"  Metrics:")
            print(f"    principal_alignment: {latest.principal_alignment:.4f}")
            print(f"    minor_alignment:     {latest.minor_alignment:.4f}")
            print(f"    subspace_angle:      {latest.subspace_angle:.4f}")
            print(f"    reconstruction_error:{latest.reconstruction_error:.4f}")

            # Show distance to thresholds
            print(f"  Distance to switch:")
            if latest.principal_alignment is not None:
                dist = self.thresholds.principal_alignment_danger - latest.principal_alignment
                status = "🟢 SAFE" if dist > 0.1 else "🟡 CLOSE" if dist > 0.05 else "🔴 DANGER"
                print(f"    Clobbering threshold: {dist:+.4f} {status}")

            if latest.minor_alignment is not None and latest.subspace_angle is not None:
                angle_ok = latest.subspace_angle < self.thresholds.subspace_angle_stable
                minor_ok = latest.minor_alignment > self.thresholds.minor_alignment_safe
                recon_ok = latest.reconstruction_error < self.thresholds.reconstruction_error_safe if latest.reconstruction_error is not None else False

                conditions_met = sum([angle_ok, minor_ok, recon_ok])
                print(f"    Optimal conditions: {conditions_met}/3 met")
                print(f"      Stable subspace:  {'✓' if angle_ok else '✗'} (angle={latest.subspace_angle:.4f} vs <0.1)")
                print(f"      Safe gradients:   {'✓' if minor_ok else '✗'} (minor={latest.minor_alignment:.4f} vs >0.6)")
                print(f"      Safe reconstruction: {'✓' if recon_ok else '✗'} (error={latest.reconstruction_error:.4f} vs <0.01)")

        print("="*80 + "\n")


def monitor_wandb(run_id: str, project: str, entity: Optional[str],
                 thresholds: SwitchThresholds, refresh_interval: int = 10):
    """Monitor metrics from WandB run."""
    try:
        import wandb
    except ImportError:
        print(f"{Colors.FAIL}Error: wandb package not installed. Install with: pip install wandb{Colors.ENDC}")
        sys.exit(1)

    api = wandb.Api()

    # Get the run
    if entity:
        run_path = f"{entity}/{project}/{run_id}"
    else:
        run_path = f"{project}/{run_id}"

    print(f"Monitoring WandB run: {run_path}")
    print(f"Refresh interval: {refresh_interval}s")
    print(f"Press Ctrl+C to stop and show summary\n")

    monitor = SVDMonitor(thresholds)
    last_step = -1

    try:
        while True:
            try:
                run = api.run(run_path)
                history = run.scan_history()

                # Process new metrics
                for row in history:
                    step = row.get('step')
                    if step is None or step <= last_step:
                        continue

                    # Extract SVD metrics for each layer
                    for key in row.keys():
                        if key.startswith('svd/') and '/mode' in key:
                            layer_name = key.replace('svd/', '').replace('/mode', '')
                            prefix = f'svd/{layer_name}/'

                            metrics = SVDMetrics(
                                step=step,
                                layer_name=layer_name,
                                mode=row.get(f'{prefix}mode', 0.0),
                                r=row.get(f'{prefix}r'),
                                principal_alignment=row.get(f'{prefix}principal_alignment'),
                                minor_alignment=row.get(f'{prefix}minor_alignment'),
                                subspace_angle=row.get(f'{prefix}subspace_angle'),
                                reconstruction_error=row.get(f'{prefix}reconstruction_error')
                            )

                            monitor.add_metrics(metrics)
                            last_step = step

                time.sleep(refresh_interval)

            except wandb.errors.CommError as e:
                print(f"{Colors.WARNING}WandB connection error: {e}. Retrying...{Colors.ENDC}")
                time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        monitor.print_summary()


def monitor_log_file(log_file: str, thresholds: SwitchThresholds, follow: bool = False):
    """Monitor metrics from training log file."""
    print(f"Monitoring log file: {log_file}")
    if follow:
        print("Following mode: waiting for new lines (Ctrl+C to stop)")
    print()

    monitor = SVDMonitor(thresholds)

    try:
        with open(log_file, 'r') as f:
            if follow:
                # Start at end of file
                f.seek(0, 2)

            while True:
                line = f.readline()
                if not line:
                    if follow:
                        time.sleep(0.5)
                        continue
                    else:
                        break

                # Parse log line for SVD metrics
                # This is a simple parser - adapt based on your log format
                if 'svd/' in line.lower():
                    # TODO: Implement log parsing based on actual log format
                    pass

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        monitor.print_summary()
    except FileNotFoundError:
        print(f"{Colors.FAIL}Error: Log file not found: {log_file}{Colors.ENDC}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor adaptive SVD training metrics in real-time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor WandB run
  python scripts/monitor_svd_metrics.py --wandb-run abc123 --project nanochat

  # Monitor with custom warning margin
  python scripts/monitor_svd_metrics.py --wandb-run abc123 --warn-margin 0.08

  # Follow log file in real-time
  python scripts/monitor_svd_metrics.py --log-file training.log --follow
        """
    )

    # Input source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--wandb-run', type=str, help='WandB run ID to monitor')
    source_group.add_argument('--log-file', type=str, help='Training log file to parse')

    # WandB options
    parser.add_argument('--project', type=str, default='nanochat', help='WandB project name')
    parser.add_argument('--entity', type=str, help='WandB entity (optional)')
    parser.add_argument('--refresh-interval', type=int, default=10, help='WandB refresh interval in seconds')

    # Log file options
    parser.add_argument('--follow', action='store_true', help='Follow log file (tail -f behavior)')

    # Threshold customization
    parser.add_argument('--warn-margin', type=float, default=0.05,
                       help='Warning margin for approaching thresholds (default: 0.05)')
    parser.add_argument('--principal-danger', type=float, default=0.4,
                       help='Principal alignment danger threshold (default: 0.4)')
    parser.add_argument('--minor-safe', type=float, default=0.6,
                       help='Minor alignment safety threshold (default: 0.6)')
    parser.add_argument('--angle-stable', type=float, default=0.1,
                       help='Subspace angle stability threshold (default: 0.1)')

    args = parser.parse_args()

    # Create thresholds
    thresholds = SwitchThresholds(
        principal_alignment_danger=args.principal_danger,
        minor_alignment_safe=args.minor_safe,
        subspace_angle_stable=args.angle_stable,
        warn_margin=args.warn_margin
    )

    # Run appropriate monitor
    if args.wandb_run:
        monitor_wandb(
            run_id=args.wandb_run,
            project=args.project,
            entity=args.entity,
            thresholds=thresholds,
            refresh_interval=args.refresh_interval
        )
    elif args.log_file:
        monitor_log_file(
            log_file=args.log_file,
            thresholds=thresholds,
            follow=args.follow
        )


if __name__ == '__main__':
    main()
