# ─── Terraform: XAI Platform AWS Infrastructure ──────────────────────────────
# Provisions: VPC, EKS cluster, RDS (PostgreSQL), ElastiCache (Redis),
#             ECR, S3 (MLflow artifacts), IAM roles, security groups

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.27"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
  }
  backend "s3" {
    bucket         = "xai-platform-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "eu-west-1"
    encrypt        = true
    dynamodb_table = "xai-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "xai-platform"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = "ml-platform-team"
    }
  }
}

# ─── Data sources ─────────────────────────────────────────────────────────────
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# ─── VPC ──────────────────────────────────────────────────────────────────────
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.5"

  name = "xai-platform-vpc"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway   = true
  single_nat_gateway   = var.environment != "production"
  enable_dns_hostnames = true
  enable_dns_support   = true

  # EKS requirements
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
    "kubernetes.io/cluster/${local.cluster_name}" = "owned"
  }
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
    "kubernetes.io/cluster/${local.cluster_name}" = "owned"
  }
}

# ─── EKS Cluster ──────────────────────────────────────────────────────────────
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.8"

  cluster_name    = local.cluster_name
  cluster_version = "1.29"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true
  cluster_endpoint_private_access = true

  cluster_addons = {
    coredns    = { most_recent = true }
    kube-proxy = { most_recent = true }
    vpc-cni    = { most_recent = true }
    aws-ebs-csi-driver = { most_recent = true }
  }

  eks_managed_node_groups = {
    general = {
      name           = "xai-general"
      instance_types = ["m5.xlarge"]
      min_size       = 2
      max_size       = 6
      desired_size   = 3
      disk_size      = 50
    }
    gpu = {
      name           = "xai-gpu"
      instance_types = ["g4dn.xlarge"]
      min_size       = 0
      max_size       = 4
      desired_size   = 0
      disk_size      = 100
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}

# ─── ECR ──────────────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "xai_platform" {
  name                 = "xai-platform"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

# ─── S3 (MLflow artifacts) ────────────────────────────────────────────────────
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "xai-platform-mlflow-${var.environment}-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

# ─── ElastiCache (Redis) ──────────────────────────────────────────────────────
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "xai-platform-redis"
  engine               = "redis"
  node_type            = "cache.t3.medium"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  engine_version       = "7.1"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.redis.name
  security_group_ids   = [aws_security_group.redis.id]
}

resource "aws_elasticache_subnet_group" "redis" {
  name       = "xai-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets
}

# ─── RDS PostgreSQL (MLflow backend) ─────────────────────────────────────────
resource "aws_db_instance" "mlflow_db" {
  identifier        = "xai-mlflow-db"
  engine            = "postgres"
  engine_version    = "16.2"
  instance_class    = "db.t3.medium"
  allocated_storage = 50
  storage_encrypted = true

  db_name  = "mlflow"
  username = "mlflow"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.mlflow.name

  backup_retention_period = 7
  skip_final_snapshot     = var.environment != "production"
  deletion_protection     = var.environment == "production"

  performance_insights_enabled = true
}

resource "aws_db_subnet_group" "mlflow" {
  name       = "xai-mlflow-db-subnet-group"
  subnet_ids = module.vpc.private_subnets
}

# ─── Security Groups ──────────────────────────────────────────────────────────
resource "aws_security_group" "redis" {
  name   = "xai-redis-sg"
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
}

resource "aws_security_group" "rds" {
  name   = "xai-rds-sg"
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
}

# ─── Locals ───────────────────────────────────────────────────────────────────
locals {
  cluster_name = "xai-platform-${var.environment}"
}

# ─── Outputs ──────────────────────────────────────────────────────────────────
output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "ecr_repository_url" {
  value = aws_ecr_repository.xai_platform.repository_url
}

output "mlflow_artifact_bucket" {
  value = aws_s3_bucket.mlflow_artifacts.id
}

output "redis_endpoint" {
  value = aws_elasticache_cluster.redis.cache_nodes[0].address
}
