import { describe, expect, it } from "vitest";
import { TestDriver } from "testdriverai/vitest/hooks";

describe("Infinite-World Research Project Page", () => {
  it("should load the project page and navigate content", async (context) => {
    const testdriver = TestDriver(context);

    // Launch Chrome to the project page
    await testdriver.provision.chrome({
      url: "https://rq-wu.github.io/projects/infinite-world",
    });

    // Scroll down through the page to view architecture and video gallery
    await testdriver.pressKeys(["escape"]);
    await testdriver.scroll("down");

    // Scroll back up to top
    await testdriver.scroll("up");
  });
});
