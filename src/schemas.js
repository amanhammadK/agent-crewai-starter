import { z } from "zod";

export const TopicSchema = z.object({
    topic: z.string().min(1, "Topic is required"),
});

export const TaskIdSchema = z.object({
    task_id: z.string().min(1, "Task ID is required"),
});
