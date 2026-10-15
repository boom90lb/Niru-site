import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const pages = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/pages" }),
  schema: z
    .object({
      title: z.string().min(1),
      description: z.string().min(1),
      eyebrow: z.string().min(1).optional(),
      headline: z.string().min(1).optional(),
      lede: z.string().min(1).optional(),
      boundary: z.string().min(1).optional(),
      lastReviewed: z.coerce.date().optional(),
      updated: z.coerce.date().optional(),
    })
    .strict(),
});

const notes = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/notes" }),
  schema: z
    .object({
      title: z.string().min(1),
      description: z.string().min(1),
      eyebrow: z.string().min(1),
      published: z.coerce.date(),
      updated: z.coerce.date().optional(),
      status: z.enum(["Method note", "Program note", "Measured study"]),
    })
    .strict(),
});

export const collections = { pages, notes };
