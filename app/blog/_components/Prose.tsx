import Image from 'next/image';
import type { ReactNode } from 'react';

// Small set of building blocks so every post has the same typography.

export const H2 = ({ children }: { children: ReactNode }) => (
    <h2 className="font-anton text-3xl md:text-4xl leading-tight mt-14 mb-5">
        {children}
    </h2>
);

export const P = ({ children }: { children: ReactNode }) => (
    <p className="text-lg leading-relaxed text-foreground/90 mb-5">
        {children}
    </p>
);

export const UL = ({ children }: { children: ReactNode }) => (
    <ul className="list-disc pl-6 space-y-2 text-lg leading-relaxed text-foreground/90 mb-5">
        {children}
    </ul>
);

export const Code = ({ children }: { children: ReactNode }) => (
    <code className="bg-background-light px-1.5 py-0.5 rounded text-[0.9em] text-primary">
        {children}
    </code>
);

export const Pre = ({ children }: { children: string }) => (
    <pre className="bg-background-light border border-border rounded-lg p-4 mb-6 overflow-x-auto text-sm leading-relaxed">
        <code>{children}</code>
    </pre>
);

export const A = ({ href, children }: { href: string; children: ReactNode }) => (
    <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-primary underline underline-offset-4 hover:opacity-80"
    >
        {children}
    </a>
);

export const Figure = ({
    src,
    alt,
    width,
    height,
    caption,
}: {
    src: string;
    alt: string;
    width: number;
    height: number;
    caption: string;
}) => (
    <figure className="my-8">
        <Image
            src={src}
            alt={alt}
            width={width}
            height={height}
            unoptimized
            className="w-full h-auto rounded-lg border border-border"
        />
        <figcaption className="text-sm text-muted-foreground mt-3">
            {caption}
        </figcaption>
    </figure>
);

export const Table = ({
    head,
    rows,
}: {
    head: string[];
    rows: string[][];
}) => (
    <div className="overflow-x-auto mb-6">
        <table className="w-full text-left text-base border-collapse">
            <thead>
                <tr className="border-b border-border">
                    {head.map((cell, index) => (
                        <th
                            key={cell}
                            className={`py-2 pr-4 font-semibold text-muted-foreground ${index ? 'text-right' : ''}`}
                        >
                            {cell}
                        </th>
                    ))}
                </tr>
            </thead>
            <tbody>
                {rows.map((row) => (
                    <tr key={row[0]} className="border-b border-border/50">
                        {row.map((cell, index) => (
                            <td
                                key={`${row[0]}-${index}`}
                                className={`py-2 pr-4 ${index ? 'text-right tabular-nums' : 'font-mono text-sm'}`}
                            >
                                {cell}
                            </td>
                        ))}
                    </tr>
                ))}
            </tbody>
        </table>
    </div>
);
