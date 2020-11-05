// <copyright file="TarFile.cs">
//  Copyright (c) 2011 Christopher A. Watford
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy of
//  this software and associated documentation files (the "Software"), to deal in
//  the Software without restriction, including without limitation the rights to
//  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
//  of the Software, and to permit persons to whom the Software is furnished to do
//  so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.
// </copyright>
// <author>Christopher A. Watford [christopher.watford@gmail.com]</author>
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace SierraEcg.IO
{
    /// <summary>
    /// Provides methods and properties to read a TAR file.
    /// </summary>
    public sealed class TarFile : IDisposable
    {
        #region Fields

        /// <summary>
        /// Block size of a TAR file.
        /// </summary>
        private const int blockSize = 512;

        private Stream innerStream;

        #endregion

        /// <summary>
        /// Gets the <see cref="Stream"/> associated with the current entry.
        /// </summary>
        public Stream Current
        {
            get;
            private set;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TarFile"/> class from
        /// the given TAR file.
        /// </summary>
        /// <param name="path">Path to a valid TAR file.</param>
        public TarFile(string path)
        {
            this.innerStream = File.OpenRead(path);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TarFile"/> class from
        /// the given stream.
        /// </summary>
        /// <param name="stream"><see cref="Stream"/> containing a TAR file.</param>
        public TarFile(Stream stream)
        {
            this.innerStream = stream;
        }

        /// <summary>
        /// Enumerates the entries in the TAR File.
        /// </summary>
        /// <param name="predicate">Optional predicate dictating which entries to read.</param>
        /// <returns>An enumeration of <see cref="TarEntry"/>s from the TAR file.</returns>
        public IEnumerable<TarEntry> EnumerateEntries(Func<TarEntry, bool> predicate = null)
        {
            byte[] block = new byte[blockSize];
            while (this.innerStream.Read(block, 0, blockSize) > 0)
            {
                if (block[0] == 0)
                    break;

                this.Current = null;
                var entry = TarEntry.FromBlock(block);
                if (predicate == null || predicate(entry))
                {
                    this.Current = new MemoryStream(entry.Size);
                }

                long position = 0;
                while (position < entry.Size)
                {
                    this.innerStream.Read(block, 0, blockSize);
                    if (this.Current != null)
                    {
                        this.Current.Write(block, 0, (int)Math.Min(entry.Size - this.Current.Position, blockSize));
                    }

                    position += blockSize;
                }

                if (this.Current != null)
                {
                    this.Current.Seek(0, SeekOrigin.Begin);
                    yield return entry;
                }
            }
        }

        /// <summary>
        /// Disposes the underlying <see cref="Stream"/>.
        /// </summary>
        public void Dispose()
        {
            this.innerStream.Dispose();
        }
    }
}
