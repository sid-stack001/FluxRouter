export type Complexity = 'simple' | 'medium' | 'complex';

export type DispatchedTier = 'fast' | 'strong' | 'cascade';

export interface Anchor {
  id: number;
  text: string;
  tier: Complexity;
  category: string;
}

export interface NearestNeighbor {
  rank: number;
  anchor: Anchor;
  similarity: number;
  isMajorityWinner: boolean;
}

export interface TierMaxScores {
  simple: number;
  medium: number;
  complex: number;
}

export interface VoteCounts {
  simple: number;
  medium: number;
  complex: number;
}

export interface CostBreakdown {
  fluxCostUsd: number;
  alwaysStrongCostUsd: number;
  savedUsd: number;
  savedPercent: number;
}

export interface RoutingResult {
  id: string;
  query: string;
  timestamp: number;
  predictedTier: Complexity;
  dispatchedTier: DispatchedTier;
  dispatchedModel: string;
  margin: number;
  marginThreshold: number;
  isDirectRouting: boolean;
  escalated: boolean;
  classifierLatencyMs: number;
  nearestNeighbors: NearestNeighbor[];
  tierMaxScores: TierMaxScores;
  voteCounts: VoteCounts;
  inputTokens: number;
  estimatedOutputTokens: number;
  cost: CostBreakdown;
  rationale: string;
}
