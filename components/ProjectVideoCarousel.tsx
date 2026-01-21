// import React, { useRef, useState, useEffect } from 'react';

// interface ProjectVideoCarouselProps {
//   demoViedo: string[];
//   images?: string[]; // Optional: if you want to show images below
// }

// const ProjectVideoCarousel: React.FC<ProjectVideoCarouselProps> = ({ demoViedo, images }) => {
//   const [current, setCurrent] = useState(0);
//   const [isPlaying, setIsPlaying] = useState(true);
//   const videoRef = useRef<HTMLVideoElement>(null);
//   const timerRef = useRef<NodeJS.Timeout | null>(null);

//   // Auto-slide to next video when current ends
//   useEffect(() => {
//     const video = videoRef.current;
//     if (!video) return;
//     const handleEnded = () => {
//       nextVideo();
//     };
//     video.addEventListener('ended', handleEnded);
//     return () => {
//       video.removeEventListener('ended', handleEnded);
//     };
//   }, [current]);

//   // Auto-play next after a fixed time (optional)
//   useEffect(() => {
//     if (!isPlaying) return;
//     timerRef.current && clearTimeout(timerRef.current);
//     timerRef.current = setTimeout(() => {
//       nextVideo();
//     }, 15000); // 15s per video (if not ended)
//     return () => {
//       timerRef.current && clearTimeout(timerRef.current);
//     };
//   }, [current, isPlaying]);

//   const nextVideo = () => {
//     setCurrent((prev) => (prev + 1) % demoViedo.length);
//   };
//   const prevVideo = () => {
//     setCurrent((prev) => (prev - 1 + demoViedo.length) % demoViedo.length);
//   };

//   return (
//     <div className="w-full flex flex-col items-center">
//       <div className="relative w-full max-w-2xl aspect-video mb-4">
//         <video
//           ref={videoRef}
//           src={demoViedo[current]}
//           controls
//           autoPlay={isPlaying}
//           muted
//           className="w-full h-full object-cover rounded-xl shadow-lg"
//           onPlay={() => setIsPlaying(true)}
//           onPause={() => setIsPlaying(false)}
//         />
//         <div className="absolute top-2 right-2 flex gap-2 z-10">
//           <button onClick={prevVideo} className="px-2 py-1 bg-black/60 text-white rounded">&#8592;</button>
//           <button onClick={nextVideo} className="px-2 py-1 bg-black/60 text-white rounded">&#8594;</button>
//         </div>
//         <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex gap-1 z-10">
//           {demoViedo.map((_, idx) => (
//             <span
//               key={idx}
//               className={`inline-block w-3 h-3 rounded-full ${idx === current ? 'bg-green-500' : 'bg-gray-400'}`}
//             />
//           ))}
//         </div>
//       </div>
//       {/* Images below video */}
//       {images && images.length > 0 && (
//         <div className="flex gap-4 flex-wrap justify-center mt-2">
//           {images.map((img, idx) => (
//             <img
//               key={idx}
//               src={img}
//               alt={`Project image ${idx + 1}`}
//               className="w-32 h-20 object-cover rounded border"
//             />
//           ))}
//         </div>
//       )}
//     </div>
//   );
// };

// export default ProjectVideoCarousel;
