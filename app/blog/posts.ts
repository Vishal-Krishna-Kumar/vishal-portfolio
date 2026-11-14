import type { ComponentType } from 'react';

import ChunkingPost, {
    meta as chunkingMeta,
} from './_posts/chunking-cuts-the-answer';
import InvertedIndexPost, {
    meta as invertedIndexMeta,
} from './_posts/inverted-index-in-numpy';

export interface IPostMeta {
    slug: string;
    title: string;
    /** ISO date, for example 2026-11-14 */
    date: string;
    summary: string;
    readingMinutes: number;
    tags: string[];
}

export interface IPost extends IPostMeta {
    Content: ComponentType;
}

// Newest first.
export const POSTS: IPost[] = [
    { ...chunkingMeta, Content: ChunkingPost },
    { ...invertedIndexMeta, Content: InvertedIndexPost },
];

export const formatDate = (iso: string) =>
    new Date(`${iso}T00:00:00Z`).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        timeZone: 'UTC',
    });
