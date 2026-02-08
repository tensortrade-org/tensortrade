/** Action/reward scheme compatibility data and helpers. */

export interface ActionSchemeInfo {
	value: string;
	label: string;
	group: "Core" | "Risk Management" | "Anti-Whipsaw" | "Position Sizing" | "Advanced";
	pbrCompatible: boolean;
}

export interface RewardSchemeInfo {
	value: string;
	label: string;
	requiresBshSemantics: boolean;
}

export const ACTION_SCHEMES: ActionSchemeInfo[] = [
	// Core
	{ value: "BSH", label: "BSH (Buy/Sell/Hold)", group: "Core", pbrCompatible: true },
	// Risk Management
	{
		value: "TrailingStopBSH",
		label: "Trailing Stop BSH",
		group: "Risk Management",
		pbrCompatible: true,
	},
	{ value: "BracketBSH", label: "Bracket BSH", group: "Risk Management", pbrCompatible: true },
	{
		value: "DrawdownBudgetBSH",
		label: "Drawdown Budget BSH",
		group: "Risk Management",
		pbrCompatible: true,
	},
	// Anti-Whipsaw
	{ value: "CooldownBSH", label: "Cooldown BSH", group: "Anti-Whipsaw", pbrCompatible: true },
	{
		value: "HoldMinimumBSH",
		label: "Hold Minimum BSH",
		group: "Anti-Whipsaw",
		pbrCompatible: true,
	},
	{
		value: "ConfirmationBSH",
		label: "Confirmation BSH",
		group: "Anti-Whipsaw",
		pbrCompatible: true,
	},
	// Position Sizing
	{
		value: "ScaledEntryBSH",
		label: "Scaled Entry BSH",
		group: "Position Sizing",
		pbrCompatible: false,
	},
	{
		value: "PartialTakeProfitBSH",
		label: "Partial Take-Profit BSH",
		group: "Position Sizing",
		pbrCompatible: false,
	},
	{
		value: "VolatilitySizedBSH",
		label: "Volatility-Sized BSH",
		group: "Position Sizing",
		pbrCompatible: true,
	},
	// Advanced
	{ value: "SimpleOrders", label: "Simple Orders", group: "Advanced", pbrCompatible: false },
	{
		value: "ManagedRiskOrders",
		label: "Managed Risk Orders",
		group: "Advanced",
		pbrCompatible: false,
	},
];

export const REWARD_SCHEMES: RewardSchemeInfo[] = [
	{ value: "SimpleProfit", label: "Simple Profit", requiresBshSemantics: false },
	{ value: "RiskAdjustedReturns", label: "Risk-Adjusted Returns", requiresBshSemantics: false },
	{ value: "PBR", label: "PBR", requiresBshSemantics: true },
	{ value: "AdvancedPBR", label: "Advanced PBR", requiresBshSemantics: true },
	{ value: "FractionalPBR", label: "Fractional PBR", requiresBshSemantics: false },
	{ value: "MaxDrawdownPenalty", label: "Max Drawdown Penalty", requiresBshSemantics: false },
];

const ACTION_MAP = new Map(ACTION_SCHEMES.map((a) => [a.value, a]));
const REWARD_MAP = new Map(REWARD_SCHEMES.map((r) => [r.value, r]));

/** Unique action scheme groups in display order. */
export const ACTION_GROUPS = [...new Set(ACTION_SCHEMES.map((a) => a.group))];

/** Check if a specific action + reward combo is valid. */
export function isCompatible(actionScheme: string, rewardScheme: string): boolean {
	const reward = REWARD_MAP.get(rewardScheme);
	if (!reward || !reward.requiresBshSemantics) return true;
	const action = ACTION_MAP.get(actionScheme);
	if (!action) return true;
	return action.pbrCompatible;
}

/** Given an action scheme, return compatible reward schemes. */
export function getCompatibleRewardSchemes(actionScheme: string): RewardSchemeInfo[] {
	const action = ACTION_MAP.get(actionScheme);
	if (!action) return REWARD_SCHEMES;
	return REWARD_SCHEMES.filter((r) => !r.requiresBshSemantics || action.pbrCompatible);
}

/** Given a reward scheme, return compatible action schemes. */
export function getCompatibleActionSchemes(rewardScheme: string): ActionSchemeInfo[] {
	const reward = REWARD_MAP.get(rewardScheme);
	if (!reward || !reward.requiresBshSemantics) return ACTION_SCHEMES;
	return ACTION_SCHEMES.filter((a) => a.pbrCompatible);
}
