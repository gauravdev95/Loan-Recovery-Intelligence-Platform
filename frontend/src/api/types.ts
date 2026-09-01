export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type RecoveryPriority =
  | "NORMAL"
  | "FOLLOW_UP"
  | "HIGH_PRIORITY"
  | "CRITICAL_PRIORITY";

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  database: string;
  version: string;
}

export interface ModelInfoResponse {
  model_name: string;
  model_version: string;
  features_version: string;
  n_features: number;
  target: string;
  loaded: boolean;
}

export interface DashboardStats {
  total_borrowers: number;
  critical_risk: number;
  high_risk: number;
  medium_risk: number;
  low_risk: number;
  avg_default_probability: number;
}

export interface BorrowerRow {
  borrower_id: string;
  default_probability: number | null;
  risk_level: RiskLevel | null;
  recovery_priority: RecoveryPriority | null;
  recommended_action?: string | null;
  model_version?: string | null;
  created_at?: string | null;
}

export interface BorrowerListResponse {
  items: BorrowerRow[];
  total: number;
  limit: number;
  offset: number;
}

export interface RecoveryQueueResponse {
  items: BorrowerRow[];
  n: number;
}

export interface BorrowerDetail extends BorrowerRow {
  features: Record<string, number | null>;
}

export interface ExplanationFactor {
  feature: string;
  value: number | null;
  shap_value: number | null;
  direction: string;
}

export interface ExplanationResponse {
  borrower_id: string | null;
  default_probability: number;
  risk_level: RiskLevel;
  recovery_priority: RecoveryPriority;
  expected_value?: number | null;
  top_factors: ExplanationFactor[];
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  pr_auc: number;
  threshold?: number;
  confusion_matrix?: { tn: number; fp: number; fn: number; tp: number };
}
