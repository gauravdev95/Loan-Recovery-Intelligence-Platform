import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { EmptyState, ErrorState, Spinner } from "./states";

describe("Spinner", () => {
  it("renders a loading label with role status", () => {
    render(<Spinner label="Loading…" />);
    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });
});

describe("ErrorState", () => {
  it("renders the message and calls onRetry", () => {
    const retry = vi.fn();
    render(<ErrorState message="boom" onRetry={retry} />);
    screen.getByRole("button", { name: /retry/i }).click();
    expect(retry).toHaveBeenCalledTimes(1);
    expect(screen.getByText("boom")).toBeInTheDocument();
  });
});

describe("EmptyState", () => {
  it("renders the empty message", () => {
    render(<EmptyState message="Nothing here" />);
    expect(screen.getByText("Nothing here")).toBeInTheDocument();
  });
});
