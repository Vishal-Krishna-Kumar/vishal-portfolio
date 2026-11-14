import type { MetadataRoute } from 'next';

import { PROJECTS } from '@/lib/data';
import { POSTS } from './blog/posts';

const SITE = 'https://www.vishalkrishna.dev';

export default function sitemap(): MetadataRoute.Sitemap {
    return [
        {
            url: SITE,
            lastModified: new Date(),
            changeFrequency: 'monthly',
            priority: 1,
        },
        {
            url: `${SITE}/blog`,
            lastModified: new Date(),
            changeFrequency: 'weekly',
            priority: 0.8,
        },
        ...POSTS.map((post) => ({
            url: `${SITE}/blog/${post.slug}`,
            lastModified: new Date(`${post.date}T00:00:00Z`),
            changeFrequency: 'yearly' as const,
            priority: 0.7,
        })),
        ...PROJECTS.map((project) => ({
            url: `${SITE}/projects/${project.slug}`,
            lastModified: new Date(),
            changeFrequency: 'monthly' as const,
            priority: 0.6,
        })),
    ];
}
