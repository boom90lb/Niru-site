import { handlePreviewRequest, type PreviewEnvironment } from "../_shared/preview-core.ts";
import { PREVIEW_TEMPORARY_ERROR_PATH } from "../../src/lib/preview-contract.ts";

interface PagesFunctionContext<Environment> {
  request: Request;
  env: Environment;
}

/**
 * Cloudflare Pages Function entrypoint. The core owns every response and never
 * falls through to a static or third-party endpoint.
 */
export async function onRequest(
  context: PagesFunctionContext<PreviewEnvironment>,
): Promise<Response> {
  try {
    return await handlePreviewRequest(context.request, context.env);
  } catch {
    const destination = new URL(PREVIEW_TEMPORARY_ERROR_PATH, context.request.url);
    return new Response(null, {
      status: 303,
      headers: {
        "Cache-Control": "no-store",
        Location: destination.toString(),
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
      },
    });
  }
}
