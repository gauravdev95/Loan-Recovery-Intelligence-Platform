import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PriorityBadge, Probability, RiskBadge } from "./badges";

describe("RiskBadge", () => {
  it("renders the risk level", () => {
    render(<RiskBadge level="CRITICAL" />);
    expect(screen.getByText("CRITICAL")).toBeInTheDocument();
  });

  it("renders a placeholder for null", () => {
    render(<RiskBadge level={null} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});

describe("PriorityBadge", () => {
  it("renders priority replacing underscores", () => {
    render(<PriorityBadge priority="HIGH_PRIORITY" />);
    expect(screen.getByText("HIGH PRIORITY")).toBeInTheDocument();
  });
});

describe("Probability", () => {
  it("formats a probability as a percentage", () => {
    render(<Probability value={0.814} />);
    expect(screen.getByText("81.4%")).toBeInTheDocument();
  });

  it("renders a placeholder for null", () => {
    render(<Probability value={null} />);
    expect(screen.getByText("—")).toBeInTheDocument();
  });
});
