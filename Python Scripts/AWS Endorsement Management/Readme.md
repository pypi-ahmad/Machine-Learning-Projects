# AWS Automation Script for AWS Endorsement Management

A collection of Python scripts using Boto3 to automate AWS resource management across EC2 and RDS services, including cleanup of snapshots, volumes, instances, and clusters.

## Overview

This is a **CLI / AWS Lambda utility** for automating AWS infrastructure management. It provides classes for EC2 and RDS resource cleanup — deleting old snapshots, removing unattached volumes, stopping/terminating untagged or idle instances, and cleaning up RDS clusters and snapshots. The scripts can run standalone or as an AWS Lambda function.

## Features

### EC2 (`ec2.py`)
- Delete EC2 snapshots older than a configurable number of days
- Delete available (unattached) EBS volumes
- Stop running EC2 instances (respects `excludepower=true` tag to skip)
- Terminate EC2 instances that lack a `user` tag
- Filter snapshots by owner ID

### RDS (`rds.py`)
- Delete RDS instance snapshots older than 2 days
- Delete RDS cluster snapshots older than 2 days
- Stop running RDS instances and clusters (respects `excludepower=true` tag)
- Delete/terminate RDS instances and clusters lacking a `user` tag
- Retain snapshots tagged with `retain=true`

### Lambda (`awsLambda.py`)
- Orchestrates EC2 and RDS cleanup across all AWS regions
- Iterates through all regions via `describe_regions()`
- Calls snapshot deletion, volume cleanup, instance shutdown, and RDS cleanup for each region

## Dependencies

> *Inferred from imports (no `requirements.txt` present)*

- `boto3`

## How It Works

1. **`awsLambda.py`** — The Lambda handler retrieves all AWS regions, then for each region instantiates `Ec2Instances` and `Rds` objects and calls their cleanup methods.
2. **`ec2.py`** — The `Ec2Instances` class connects to EC2 via Boto3. `delete_snapshots()` removes snapshots older than N days. `delete_available_volumes()` deletes unattached EBS volumes. `shutdown()` stops running instances (skipping those tagged `excludepower=true`) and terminates instances without a `user` tag.
3. **`rds.py`** — The `Rds` class connects to RDS via Boto3. `cleanup_snapshot()` deletes instance and cluster snapshots older than 2 days (skipping those tagged `retain=true`). `cleanup_instances()` stops or deletes RDS clusters and instances based on their tags.

## Project Structure

```
AWS Automation Script for AWS endorsement management/
├── awsLambda.py    # AWS Lambda handler orchestrating multi-region cleanup
├── ec2.py          # EC2 resource management (snapshots, volumes, instances)
├── rds.py          # RDS resource management (snapshots, clusters, instances)
└── Readme.md       # This file
```

## Setup & Installation

```bash
pip install boto3
```

Configure AWS credentials via one of:
- AWS CLI: `aws configure`
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- AWS IAM role (for Lambda execution)

## How to Run

### Standalone (EC2):
```bash
cd "AWS Automation Script for AWS endorsement management"
python ec2.py
```
This runs EC2 cleanup in `us-east-1` (deletes snapshots older than 3 days and shuts down instances).

### Standalone (RDS):
```bash
python rds.py
```
This runs RDS snapshot and instance cleanup in `us-east-1`.

### As AWS Lambda:
Deploy `awsLambda.py` (along with `ec2.py` and `rds.py`) as a Lambda function. The `lambda_handler` function serves as the entry point.

## Configuration

- **Owner ID:** In `ec2.py`, `get_user_created_snapshots()` filters by a hardcoded owner ID (`'your owner id'`) — must be replaced with your actual AWS account ID.
- **Snapshot age threshold:** Passed as a parameter to `delete_snapshots()` (1 day in Lambda, 3 days in standalone `ec2.py`, default parameter: 2 days).
- **RDS snapshot age:** Hardcoded to 2 days in `rds.py` `_is_older_snapshot()`.
- **Tag-based exclusions:**
  - `excludepower=true` — prevents stopping an instance
  - `user` tag — presence prevents termination/deletion
  - `retain=true` — prevents snapshot deletion (RDS)
- **AWS region:** Standalone scripts default to `us-east-1`; the Lambda handler iterates all regions.

## Testing

No formal test suite present.

## Limitations

- The `get_user_created_snapshots()` method in `ec2.py` has a placeholder owner ID (`'your owner id'`) that must be replaced.
- In `ec2.py`, `delete_snapshots()` calls `get_nimesa_created_snapshots()` which is not defined — it should likely call `get_user_created_snapshots()`.
- The snapshot deletion counter in `ec2.py` has a bug: `delete_snapshots_num + 1` should be `delete_snapshots_num += 1`.
- In `rds.py`, `_delete_instance()` and `_delete_cluster()` call `describe_*` instead of `delete_*` — they do not actually delete resources.
- In `rds.py`, `_can_delete_instance()` checks if `'user'` is `in tag` (checking each tag dict) rather than checking the tag's `Key` field specifically.
- No dry-run mode — all operations are destructive.
- Error handling is minimal (bare `except Exception` blocks with print statements).

## Security Notes

- AWS credentials are expected to be configured externally (AWS CLI, environment variables, or IAM role). The code includes a commented-out section showing access key / secret key usage — do not hardcode credentials.
- The scripts perform destructive operations (deleting snapshots, terminating instances) — ensure proper IAM permissions and test thoroughly.
